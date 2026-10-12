# services/job_recommender.py
#
# Sources: Adzuna (India, primary) + Remotive (remote backup).
# Pipeline: fetch -> normalise -> filter (link / freshness / role relevance /
# experience level / location + remote eligibility) -> rank -> de-duplicate
# -> verify links -> return.
#
# Ranking priority (highest first):
#   1. location tier   (selected city beats remote-eligible)
#   2. experience tier (internship / entry-level beat "not specified")
#   3. match score     (role + skills + small location/level bonus)

import os
import re
import html
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ADZUNA_URL = "https://api.adzuna.com/v1/api/jobs/in/search/1"

PIPELINE_VERSION = "relevance-v2"   # shown in JOB_DEBUG output / trace
REQUEST_TIMEOUT = 15
LINK_CHECK_TIMEOUT = 8
MAX_JOB_AGE_DAYS = 60
SOURCE_CACHE_TTL = 15 * 60      # the dashboard refetches on every window focus
LINK_CACHE_TTL = 30 * 60
MAX_LINK_CHECKS = 30
DEFAULT_TOP_N = 10

STOP_WORDS = {
    "and", "the", "to", "of", "a", "an", "in", "on", "for",
    "with", "or", "at", "by", "as", "is", "are", "be", "from",
    "that", "this", "it", "you", "your", "we", "our", "will",
    "can", "have", "has", "not", "but", "if", "about", "into",
    "job", "jobs", "role", "roles", "work", "working", "team",
    "company", "experience", "years", "year", "new", "skills",
    "skill", "knowledge", "required", "requirements", "remote",
    "full", "time", "part", "position",
}

# Words in a target role that describe level, not the profession.
ROLE_NOISE = STOP_WORDS | {
    "intern", "interns", "internship", "fresher", "freshers", "junior",
    "trainee", "graduate", "entry", "level", "senior",
}

SKILL_ALIASES = {
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "neural networks"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "natural language processing": ["natural language processing", "nlp"],
    "data science": ["data science", "data scientist"],
    "data analysis": ["data analysis", "data analyst", "data analytics"],
    "data visualization": [
        "data visualization", "data visualisation",
        "tableau", "power bi", "matplotlib"
    ],
    "javascript": ["javascript", "js"],
    "react": ["react", "react.js", "reactjs"],
    "node.js": ["node.js", "nodejs"],
    "python": ["python"],
    "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "tensorflow": ["tensorflow", "keras"],
    "pytorch": ["pytorch"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "power bi": ["power bi", "powerbi"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp", "csharp"],
}

# A selected city matches a listing only if one of these names appears in the
# listing's own location data. No state/region fallback: Pune must not return
# Mumbai just because both are in Maharashtra.
CITY_ALIASES = {
    "mumbai": ["mumbai", "navi mumbai", "bombay"],
    "pune": ["pune"],
    "bengaluru": ["bengaluru", "bangalore"],
    "bangalore": ["bengaluru", "bangalore"],
    "delhi": ["delhi", "new delhi", "delhi ncr"],
    "new delhi": ["delhi", "new delhi", "delhi ncr"],
    "hyderabad": ["hyderabad", "secunderabad"],
    "chennai": ["chennai", "madras"],
    "ahmedabad": ["ahmedabad"],
    "kolkata": ["kolkata", "calcutta"],
}

GLOBAL_LOCATION_CHOICES = {
    "", "worldwide", "anywhere", "global", "remote", "any",
    "worldwide / remote",
}
INDIA_LOCATION_CHOICES = {"india", "anywhere in india"}

# ---------------------------------------------------------------------------
# Experience-level patterns (applied to normalised, lower-case text)
# ---------------------------------------------------------------------------

_SENIOR_TITLE = re.compile(
    r"\b(senior|sr|principal|director|vp|vice president|head of|chief|"
    r"architect|expert|staff engineer)\b|\blead\b(?!\s+gen)"
)
_MANAGER_TITLE = re.compile(r"\bmanager\b")
_INTERN_TITLE = re.compile(r"\b(intern|interns|internship|internships)\b")
_ENTRY_TITLE = re.compile(
    r"\b(fresher|freshers|entry[- ]level|graduate|graduates|junior|jr|"
    r"trainee|trainees|early career)\b"
)
_SENIOR_DESC = re.compile(r"\bsenior[- ]level\b")
_ENTRY_DESC = re.compile(
    r"\bfreshers?\b|\bentry[- ]level\b|\brecent graduates?\b|"
    r"\bno (?:prior |previous )?experience (?:is )?(?:required|needed|necessary)\b"
)
_YEARS_BEFORE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|to)\s*\d{1,2}\s*\+?\s*)?"
    r"(?:years?|yrs?)\b(?:\s+of)?(?:\s+\w+){0,3}?\s+(?:experience|exp)\b"
)
_YEARS_AFTER = re.compile(
    r"experience\s*(?:of|:|-)?\s*(?:at least\s*|minimum\s*(?:of\s*)?)?"
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|to)\s*\d{1,2})?\s*(?:years?|yrs?)"
)
_REMOTE_MARKER = re.compile(r"\b(remote|work from home|work-from-home|wfh)\b")
_INDIA_EXCLUSION = re.compile(
    r"\b(?:not|except|excluding|exclude|outside|excl)\b[^,;/|]{0,20}\bindia\b"
)
_INDIA_OR_GLOBAL = re.compile(
    r"\bindia\b|\bworldwide\b|\bworld wide\b|\bglobal(?:ly)?\b|"
    r"\banywhere\b(?!\s+in\s+(?!india))"
)

GENERIC_PATHS = {
    "", "/", "/jobs", "/remote-jobs", "/careers", "/search",
    "/job-search", "/login", "/signin",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def strip_html(value):
    if not value:
        return ""
    value = str(value)
    value = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ", value, flags=re.S | re.I
    )
    value = re.sub(r"<[^>]+>", " ", value)
    return html.unescape(value)


def normalize_text(value):
    if not value:
        return ""
    value = strip_html(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9+#./\- ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _contains_term(text, term):
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return bool(re.search(pattern, text))


def _stems(text):
    return {w[:5] for w in re.split(r"[^a-z0-9+#]+", text) if w}


def clean_skill_list(skills):
    if isinstance(skills, str):
        skills = skills.split(",")
    if not isinstance(skills, list):
        return []

    output = []
    seen = set()

    for skill in skills:
        label = str(skill).strip()
        key = normalize_text(label)
        if key and key not in seen and key not in STOP_WORDS:
            output.append(label)
            seen.add(key)

    return output


def skill_variants(skill):
    key = normalize_text(skill)
    return [
        normalize_text(value)
        for value in SKILL_ALIASES.get(key, [key])
        if value
    ]


def skill_matches(skill, job_text):
    text = normalize_text(job_text)
    return any(
        _contains_term(text, variant) for variant in skill_variants(skill)
    )


def _safe_log_error(source, exc):
    """Log without the exception message: requests embeds the full URL
    (including app_key) in HTTPError text."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    suffix = f" (HTTP {status})" if status else ""
    print(f"{source} API error: {type(exc).__name__}{suffix}")


# ---------------------------------------------------------------------------
# Tiny in-memory TTL cache
# ---------------------------------------------------------------------------

_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _cache_get(key, ttl):
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
    if entry and time.time() - entry[0] < ttl:
        return entry[1]
    return None


def _cache_put(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Location logic
# ---------------------------------------------------------------------------

def is_global_location(location):
    return normalize_text(location) in GLOBAL_LOCATION_CHOICES


def primary_location(location):
    """'Pune, Maharashtra' -> 'Pune'. Dropdown values pass through unchanged."""
    text = str(location or "").strip()
    if normalize_text(text) in GLOBAL_LOCATION_CHOICES | INDIA_LOCATION_CHOICES:
        return text
    return text.split(",")[0].strip()


def location_mode(location):
    key = normalize_text(primary_location(location))
    if key in GLOBAL_LOCATION_CHOICES:
        return "global"
    if key in INDIA_LOCATION_CHOICES:
        return "india"
    return "city"


def city_terms(location):
    key = normalize_text(primary_location(location))
    return CITY_ALIASES.get(key, [key])


def remote_eligible(candidate_location):
    """True only when the listing explicitly says India or worldwide."""
    text = normalize_text(candidate_location)
    if not text or _INDIA_EXCLUSION.search(text):
        return False
    return bool(_INDIA_OR_GLOBAL.search(text))


def _adzuna_is_countrywide(job):
    area = [normalize_text(a) for a in (job.get("area") or [])]
    name = normalize_text(job.get("candidate_required_location", ""))
    return all(a == "india" for a in area) and name in ("", "india")


def classify_location(job, mode, terms):
    """Return {tier, kind, remote, note} or None when the job must be dropped."""
    location_text = normalize_text(" ".join(
        [str(job.get("candidate_required_location", "") or "")]
        + [str(a) for a in (job.get("area") or [])]
    ))
    shown = str(job.get("candidate_required_location", "") or "").strip()
    city_hit = mode == "city" and any(
        _contains_term(location_text, t) for t in terms
    )

    if job.get("source") == "Remotive":
        # Every Remotive job is remote: keep it only if India/worldwide is explicit.
        if not remote_eligible(shown):
            return None
        return {
            "tier": 0 if city_hit else (1 if mode == "city" else 0),
            "kind": "city" if city_hit else "remote_eligible",
            "remote": True,
            "note": f"Remote; listing says: {shown}",
        }

    # Adzuna (India endpoint): location comes from the listing itself.
    if mode == "city":
        if city_hit:
            return {"tier": 0, "kind": "city", "remote": False,
                    "note": f"Listed in {shown}"}
        marker = _REMOTE_MARKER.search(normalize_text(
            f"{job.get('title', '')} {job.get('description', '')}"
        ))
        if marker and _adzuna_is_countrywide(job):
            return {"tier": 1, "kind": "remote_eligible", "remote": True,
                    "note": "Listing mentions remote/work-from-home; "
                            "location is India-wide"}
        return None

    return {
        "tier": 0,
        "kind": "india",
        "remote": bool(_REMOTE_MARKER.search(normalize_text(
            f"{job.get('title', '')} {job.get('description', '')}"
        ))),
        "note": f"Listed in {shown}" if shown else "Location not specified",
    }


# ---------------------------------------------------------------------------
# Experience level
# ---------------------------------------------------------------------------

def _min_years(text):
    values = [int(m.group(1)) for m in _YEARS_BEFORE.finditer(text)]
    values += [int(m.group(1)) for m in _YEARS_AFTER.finditer(text)]
    return min(values) if values else None


def classify_experience(job):
    """Return {"level": internship|entry|senior|unspecified, "evidence": str}."""
    title = normalize_text(job.get("title", ""))
    description = normalize_text(job.get("description", ""))
    job_type = normalize_text(job.get("job_type", ""))

    match = _SENIOR_TITLE.search(title)
    if match:
        return {"level": "senior", "evidence": f"title contains '{match.group(0).strip()}'"}

    if _INTERN_TITLE.search(title):
        return {"level": "internship", "evidence": "title says internship"}
    if re.search(r"\bintern", job_type):
        return {"level": "internship", "evidence": "listing type is internship"}

    match = _ENTRY_TITLE.search(title)
    if match:
        return {"level": "entry", "evidence": f"title contains '{match.group(0)}'"}

    if _MANAGER_TITLE.search(title):
        return {"level": "senior", "evidence": "title contains 'manager'"}

    years = _min_years(description)
    if years is not None and years >= 3:
        return {"level": "senior",
                "evidence": f"description asks for {years}+ years of experience"}
    if _SENIOR_DESC.search(description):
        return {"level": "senior", "evidence": "description says senior-level"}

    if years is not None and years <= 1:
        return {"level": "entry",
                "evidence": f"description asks for {years} years of experience"}
    match = _ENTRY_DESC.search(description)
    if match:
        return {"level": "entry", "evidence": f"description says '{match.group(0)}'"}

    return {"level": "unspecified", "evidence": ""}


EXPERIENCE_LABELS = {
    "internship": "Internship",
    "entry": "Entry Level",
    "senior": "Senior / Experienced",
    "unspecified": "Not specified",
}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Role relevance (applied FIRST, on the job TITLE only)
#
# A job is relevant when its title contains a term that defines the target
# role. Skills and description text never make a job relevant, so a listing
# that merely mentions Python / AI / SQL cannot slip through on a skill match.
#
# Known role families use explicit title-term lists. Any other role (e.g.
# "Software Engineer") must appear in the title as an ordered phrase, so a
# user who asks for it still gets it, and nobody else does.
# ---------------------------------------------------------------------------

_ML_TERMS = [
    r"\bmachine learning\b", r"\bml\b", r"\bdeep learning\b", r"\bnlp\b",
    r"\bnatural language processing\b", r"\bcomputer vision\b",
    # "AI Intern" yes; "AI Code Trainer" no (trainee/intern/research only).
    r"\b(?:ai|artificial intelligence)\s+(?:intern|interns|internship|"
    r"trainee|research|researcher)\b",
]
_DATA_SCIENCE_TERMS = [
    r"\bdata scien\w*", r"\bapplied scientist\b",
    r"\bdata\s+(?:intern|interns|internship|trainee|associate)\b",
]
_ANALYTICS_CORE = [
    r"\bdata analy\w*", r"\banalytics\b", r"\bbusiness intelligence\b",
    r"\bbi\s+(?:analyst|developer|intern|internship|engineer)\b",
    r"\bstatistic\w*",
]
_ANALYST_EXTRA = [
    r"\bbusiness analy\w*", r"\breporting analy\w*", r"\binsights?\s+analy\w*",
    r"\bsql\b", r"\bpower bi\b", r"\btableau\b",
]

ROLE_FAMILIES = {
    "data science": _DATA_SCIENCE_TERMS + _ML_TERMS + _ANALYTICS_CORE,
    "machine learning": _ML_TERMS + _DATA_SCIENCE_TERMS,
    "data analytics": _ANALYTICS_CORE + _ANALYST_EXTRA,
}
# (pattern on the normalised role alternative, family). First match wins.
ROLE_FAMILY_RULES = [
    (r"\bdata scien", "data science"),
    (r"\b(?:machine learning|deep learning|nlp|ml)\b", "machine learning"),
    (r"\b(?:data analy|business analy|business intelligence|bi|analytics|"
     r"reporting analy)", "data analytics"),
]
_FAMILY_REGEX = {
    name: re.compile("|".join(patterns)) for name, patterns in ROLE_FAMILIES.items()
}
# split alternatives like "Data Scientist / ML Engineer", but keep "AI/ML" intact
_ROLE_SPLIT = re.compile(r"\s+/\s+|\s*[,|;]\s*|\s+&\s+|\s+or\s+")
_STEM_ALIASES = {"devel": "engin", "progr": "engin"}   # developer == engineer


def _phrase_stem(word):
    stem = word[:5]
    return _STEM_ALIASES.get(stem, stem)


def _ordered_phrase_in_title(role_stems, title_tokens):
    """role words must appear in order, with at most one word between them."""
    for start, token in enumerate(title_tokens):
        if token != role_stems[0]:
            continue
        position = start
        for stem in role_stems[1:]:
            nxt = next((j for j in (position + 1, position + 2)
                        if j < len(title_tokens) and title_tokens[j] == stem), None)
            if nxt is None:
                break
            position = nxt
        else:
            return True
    return False


def role_relevance(target_role, title):
    """Return (relevant, strength, reason). Title only; never uses skills or
    description. strength is 'exact' or 'family'/'phrase'."""
    title = normalize_text(title)
    role_full = normalize_text(target_role)
    if not role_full or not title:
        return False, "", "no role or title"
    if role_full in title:
        return True, "exact", f"title contains '{role_full}'"

    title_tokens = [_phrase_stem(w) for w in re.split(r"[^a-z0-9+#]+", title) if w]

    for alternative in _ROLE_SPLIT.split(str(target_role)):
        alt = normalize_text(alternative)
        words = [w for w in alt.split() if w not in ROLE_NOISE]
        if not alt or not words:
            continue
        if alt in title:
            return True, "exact", f"title contains '{alt}'"

        family = next((name for pattern, name in ROLE_FAMILY_RULES
                       if re.search(pattern, alt)), None)
        if family:
            match = _FAMILY_REGEX[family].search(title)
            if match:
                return True, "family", (
                    f"title has {family} term '{match.group(0).strip()}'")
            continue

        stems = [_phrase_stem(w) for w in words]
        if _ordered_phrase_in_title(stems, title_tokens):
            return True, "phrase", f"title contains role phrase '{' '.join(words)}'"

    return False, "", f"title has no term defining the role '{role_full}'"


def calculate_match(target_role, user_skills, job, level="unspecified",
                    location_kind="india"):
    """Role (40) + skills (35) + location (10) + level (15) = 100 max.

    This score is informational. Ranking applies location tier and
    experience tier first, so a high skill score cannot outrank eligibility.
    """
    title = normalize_text(job.get("title", ""))
    job_text = " ".join([
        title,
        normalize_text(job.get("description", "")),
        normalize_text(job.get("category", "")),
        normalize_text(" ".join(job.get("tags", []) or [])),
    ])

    relevant, strength, _ = role_relevance(target_role, title)
    role_score = (40 if strength == "exact" else 35) if relevant else 0
    matched = [s for s in user_skills if skill_matches(s, job_text)]
    skill_score = len(matched) / len(user_skills) * 35 if user_skills else 0
    location_score = {"city": 10, "remote_eligible": 7}.get(location_kind, 5)
    level_score = 15 if level in ("internship", "entry") else 5

    total = role_score + skill_score + location_score + level_score
    return round(min(total, 100)), matched


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def is_job_specific_url(url):
    """Structural check only: never turns a homepage into an apply link."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    path = parsed.path.rstrip("/").lower() or "/"
    return not (path in GENERIC_PATHS and not parsed.query)


def _url_key(url):
    p = urlparse(url)
    return f"{p.netloc.lower()}{p.path.rstrip('/')}"


def check_link(url):
    """Return 'ok', 'dead' or 'unknown' (e.g. site blocks bots / times out).

    'dead' = 404/410, or the link redirects to a homepage / generic page.
    The original URL is never replaced by the redirect target.
    """
    cached = _cache_get(("link", url), LINK_CACHE_TTL)
    if cached:
        return cached

    result = "unknown"
    try:
        with requests.get(
            url,
            stream=True,
            allow_redirects=True,
            timeout=LINK_CHECK_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; InternTrustLinkCheck/1.0)"},
        ) as response:
            status = response.status_code
            final = urlparse(response.url)
            final_path = final.path.rstrip("/").lower() or "/"
            original_path = urlparse(url).path.rstrip("/").lower() or "/"

            if status in (404, 410):
                result = "dead"
            elif (
                status < 400
                and final_path in GENERIC_PATHS
                and not final.query
                and original_path not in GENERIC_PATHS
            ):
                result = "dead"          # redirected to a homepage
            elif status < 400:
                result = "ok"
    except requests.RequestException:
        result = "unknown"

    _cache_put(("link", url), result)
    return result


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _fetch_adzuna_page(app_id, app_key, query, city):
    cache_key = ("adzuna", query, city)
    cached = _cache_get(cache_key, SOURCE_CACHE_TTL)
    if cached is not None:
        return cached

    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 20,
        "what": query,
        "max_days_old": MAX_JOB_AGE_DAYS,
        "content-type": "application/json",
    }
    if city:
        params["where"] = city

    try:
        response = requests.get(
            ADZUNA_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        results = response.json().get("results", []) or []
    except (requests.RequestException, ValueError) as exc:
        _safe_log_error("Adzuna", exc)
        return []

    _cache_put(cache_key, results)
    return results


def fetch_adzuna_jobs(target_role, skills, location):
    """Fetch Indian listings from Adzuna using server-side credentials.

    A selected city is searched on its own (no state / India-wide fallback),
    so results can't drift to unrelated cities. Each listing keeps Adzuna's
    own redirect_url and location data.
    """
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        print("Adzuna credentials are missing (set ADZUNA_APP_ID / ADZUNA_APP_KEY in .env).")
        return []

    mode = location_mode(location)
    city = primary_location(location) if mode == "city" else ""

    role = target_role.strip()
    queries = [role, f"{role} intern", f"{role} fresher"]
    for skill in skills[:2]:
        queries.append(f"{role} {skill}")
    queries = list(dict.fromkeys(queries))

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(
            lambda q: _fetch_adzuna_page(app_id, app_key, q, city), queries
        ))

    jobs = []
    seen = set()
    raw_count = 0

    for results in pages:
        for item in results:
            raw_count += 1
            link = str(item.get("redirect_url", "") or "").strip()
            title = str(item.get("title", "") or "").strip()
            ident = str(item.get("id", "") or "")

            if not title or not link or (ident and ident in seen):
                continue
            if ident:
                seen.add(ident)

            company_data = item.get("company") or {}
            location_data = item.get("location") or {}
            category_data = item.get("category") or {}

            contract_time = str(item.get("contract_time", "") or "")
            contract_type = str(item.get("contract_type", "") or "")

            jobs.append({
                "id": ident,
                "title": title,
                "company_name": company_data.get("display_name", ""),
                "url": link,                              # untouched
                # Only what Adzuna returns: no fallback to the searched city.
                "candidate_required_location":
                    location_data.get("display_name", "") or "",
                "area": location_data.get("area") or [],
                "description": strip_html(item.get("description", "")),
                "category": category_data.get("label", ""),
                "tags": [],
                "job_type": " ".join(
                    p for p in [contract_time, contract_type] if p
                ),
                "publication_date": item.get("created", ""),
                "source": "Adzuna",
            })

    print(f"Adzuna: {len(queries)} request(s), {raw_count} listings, "
          f"{len(jobs)} unique (where={city or 'India-wide'}).")
    return jobs


def fetch_remotive(search_term, limit=50):
    cache_key = ("remotive", search_term, limit)
    cached = _cache_get(cache_key, SOURCE_CACHE_TTL)
    if cached is not None:
        return cached

    params = {"limit": limit}
    if search_term:
        params["search"] = search_term

    try:
        response = requests.get(
            REMOTIVE_URL, params=params, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        jobs = response.json().get("jobs", []) or []
    except (requests.RequestException, ValueError) as exc:
        _safe_log_error("Remotive", exc)
        return []

    _cache_put(cache_key, jobs)
    return jobs


def collect_remotive_jobs(target_role, skills, location):
    """Keep Remotive jobs only when the listing explicitly names India or
    worldwide. Applies to every location choice, not just city searches."""
    queries = [target_role, f"{target_role} intern"]
    for skill in skills[:2]:
        queries.append(f"{target_role} {skill}")

    queries = list(dict.fromkeys(queries))
    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(fetch_remotive, queries))

    jobs = []
    seen = set()

    for page in pages:
        for job in page:
            link = str(job.get("url", "") or "").strip()   # job-specific, untouched
            if not link or link in seen:
                continue
            seen.add(link)

            if not remote_eligible(job.get("candidate_required_location", "")):
                continue

            item = dict(job)
            item["source"] = "Remotive"
            jobs.append(item)

    return jobs


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _is_stale(date_value):
    if not date_value:
        return False
    try:
        parsed = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - parsed > timedelta(days=MAX_JOB_AGE_DAYS)


def _drop(ctx, stats, key, job, detail=""):
    stats[key] += 1
    ctx["drops"].append({
        "title": str(job.get("title", "") or ""),
        "company": str(job.get("company_name", job.get("company", "")) or ""),
        "source": job.get("source", ""),
        "link": str(job.get("url", "") or ""),
        "reason": key,
        "detail": detail,
    })
    return None


def _build_candidate(job, ctx, stats):
    title = str(job.get("title", "") or "").strip()
    link = str(job.get("url", "") or "").strip()
    company = str(job.get("company_name", job.get("company", "")) or "").strip()

    # 1. Role relevance FIRST: title only, before anything is ranked or checked.
    relevant, strength, relevance_reason = role_relevance(ctx["role"], title)
    if not relevant:
        return _drop(ctx, stats, "irrelevant_role", job, relevance_reason)

    # 2. Link structure / freshness (network link checks happen much later).
    if not title or not is_job_specific_url(link):
        return _drop(ctx, stats, "invalid_link", job, "missing or generic URL")
    if _is_stale(job.get("publication_date")):
        return _drop(ctx, stats, "stale", job, str(job.get("publication_date")))

    description = strip_html(job.get("description", ""))
    scoring_job = dict(job)
    scoring_job["description"] = description
    title_norm = normalize_text(title)

    # 3. Experience level
    experience = classify_experience(scoring_job)
    if experience["level"] == "senior" and not ctx["include_senior"]:
        return _drop(ctx, stats, "senior", job, experience["evidence"])

    # 4. Location / remote eligibility
    place = classify_location(scoring_job, ctx["mode"], ctx["terms"])
    if place is None:
        return _drop(ctx, stats, "location_or_eligibility", job,
                     str(job.get("candidate_required_location", "")))

    score, matched = calculate_match(
        ctx["role"], ctx["skills"], scoring_job,
        level=experience["level"], location_kind=place["kind"],
    )

    if ctx["include_senior"]:
        exp_tier = 0
    else:
        exp_tier = 0 if experience["level"] in ("internship", "entry") else 1

    shown_location = str(job.get("candidate_required_location", "") or "").strip()

    return {
        "title": title,
        "company": company or "Company",
        "display_title": f"{title} at {company}" if company else title,
        "location": shown_location or "Location not specified",
        "type": str(job.get("job_type", "") or "Not specified")
            .replace("_", " ").title(),
        "category": job.get("category", ""),
        "experience_level": EXPERIENCE_LABELS[experience["level"]],
        "experience_evidence": experience["evidence"],
        "relevance_reason": relevance_reason,
        "remote": place["remote"],
        "eligibility": place["note"],
        "match": score,
        "matched_skills": matched,
        "published": job.get("publication_date", ""),
        "link": link,
        "source": job.get("source", ""),
        "_rank": (place["tier"], exp_tier, -score, -len(matched)),
        "_ids": {
            ("url", _url_key(link)),
            ("fp", f"{title_norm}|{normalize_text(company)}|"
                   f"{normalize_text(shown_location)}"),
            *([("id", job.get("source"), str(job["id"]))] if job.get("id") else []),
        },
    }


def _select_verified(candidates, top_n, stats):
    if os.getenv("JOB_LINK_VERIFY", "1") == "0":
        for c in candidates[:top_n]:
            c["link_status"] = "unverified"
        return candidates[:top_n]

    selected = []
    index = checked = 0

    while (
        index < len(candidates)
        and len(selected) < top_n
        and checked < MAX_LINK_CHECKS
    ):
        batch = candidates[index:index + max(top_n - len(selected), 4)]
        index += len(batch)
        checked += len(batch)

        with ThreadPoolExecutor(max_workers=min(8, len(batch))) as pool:
            statuses = list(pool.map(check_link, [c["link"] for c in batch]))

        for candidate, status in zip(batch, statuses):
            if status == "dead":
                stats["dead_link"] += 1
                continue
            candidate["link_status"] = "verified" if status == "ok" else "unverified"
            selected.append(candidate)

    return selected[:top_n]


def get_safe_jobs_detailed(user_input, top_n=DEFAULT_TOP_N):
    """Return {"jobs": [...], "stats": {...}} (stats explain every drop)."""
    user_input = user_input or {}

    target_role = str(user_input.get("target_role", "") or "").strip()
    user_skills = clean_skill_list(user_input.get("skills", []))
    location = str(user_input.get("location", "") or "").strip()

    stats = {k: 0 for k in (
        "invalid_link", "stale", "irrelevant_role", "senior",
        "location_or_eligibility", "duplicate", "dead_link",
    )}

    if not target_role:
        return {"jobs": [], "stats": stats}

    role_norm = normalize_text(target_role)
    ctx = {
        "role": target_role,
        "skills": user_skills,
        "mode": location_mode(location),
        "terms": city_terms(location),
        "drops": [],
        # Senior roles only when the user explicitly asks for them.
        "include_senior": bool(user_input.get("include_senior"))
            or bool(_SENIOR_TITLE.search(role_norm))
            or bool(_MANAGER_TITLE.search(role_norm)),
    }

    adzuna_jobs = fetch_adzuna_jobs(target_role, user_skills, location)
    remotive_jobs = collect_remotive_jobs(target_role, user_skills, location)
    print(f"Sources: adzuna={len(adzuna_jobs)} remotive_eligible={len(remotive_jobs)}")

    candidates = []
    for job in adzuna_jobs + remotive_jobs:
        candidate = _build_candidate(job, ctx, stats)
        if candidate:
            candidates.append(candidate)

    candidates.sort(key=lambda c: c["_rank"])

    unique, seen = [], set()
    for candidate in candidates:
        if candidate["_ids"] & seen:
            stats["duplicate"] += 1
            continue
        seen |= candidate["_ids"]
        unique.append(candidate)

    jobs = _select_verified(unique, top_n, stats)
    for job in jobs:
        job.pop("_rank", None)
        job.pop("_ids", None)

    trace = {
        "version": PIPELINE_VERSION,
        "input": {
            "target_role": target_role,
            "skills": user_skills,
            "location": location,
            "mode": ctx["mode"],
        },
        "raw": {"adzuna": len(adzuna_jobs), "remotive_eligible": len(remotive_jobs)},
        "dropped": ctx["drops"],
    }
    if os.getenv("JOB_DEBUG") == "1":
        _print_debug(trace, jobs)

    return {"jobs": jobs, "stats": stats, "trace": trace}


def _print_debug(trace, jobs):
    """Terminal trace for diagnosing 'why is this job here?'. No secrets."""
    i = trace["input"]
    print(f"[JOB_DEBUG {trace['version']}] role={i['target_role']!r} "
          f"skills={i['skills']} location={i['location']!r} mode={i['mode']}")
    print(f"[JOB_DEBUG] raw after source fetch: {trace['raw']}")
    for d in trace["dropped"]:
        print(f"[JOB_DEBUG] DROP {d['reason']:<24} {d['source']:<8} "
              f"{d['title'][:60]!r} ({d['detail'][:60]})")
    for n, j in enumerate(jobs, 1):
        print(f"[JOB_DEBUG] KEEP #{n} {j['source']:<8} {j['title'][:60]!r} "
              f"<- {j['relevance_reason']}")


def get_safe_jobs(user_input, top_n=DEFAULT_TOP_N):
    return get_safe_jobs_detailed(user_input, top_n)["jobs"]