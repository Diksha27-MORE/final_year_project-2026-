#!/usr/bin/env python
"""
Live smoke test for InternTrust job recommendations (real Adzuna + Remotive).

Run from the backend folder (the one containing app.py and .env):
    python scripts/live_check.py
    python scripts/live_check.py --location Mumbai --role "Data Scientist"
    python scripts/live_check.py --http http://localhost:5000 --email you@example.com

Secrets: ADZUNA_APP_ID / ADZUNA_APP_KEY are read from .env and are only ever
reported as "set (N chars)". Every line printed passes through redact().
Exit code is 1 if any FAIL line was printed.
"""
import argparse
import getpass
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
except ImportError:
    print("python-dotenv not installed; relying on real environment variables.")

from services import job_recommender as jr  # noqa: E402

FAILS = []


def redact(text):
    for key in ("ADZUNA_APP_ID", "ADZUNA_APP_KEY"):
        value = os.getenv(key)
        if value and len(value) >= 4:
            text = text.replace(value, "***")
    return text


def p(*parts):
    print(redact(" ".join(str(x) for x in parts)))


def ok(msg):
    p("  PASS ", msg)


def warn(msg):
    p("  WARN ", msg)


def fail(msg):
    FAILS.append(msg)
    p("  FAIL ", msg)


def head(title):
    p("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def short(value, n):
    value = str(value or "")
    return value if len(value) <= n else value[: n - 1] + "…"


# --------------------------------------------------------------------- 1. env
def step_env():
    head("1. Environment")
    for key in ("ADZUNA_APP_ID", "ADZUNA_APP_KEY"):
        value = os.getenv(key)
        if value:
            ok(f"{key} is set ({len(value)} chars)")
        else:
            fail(f"{key} is missing from {BACKEND / '.env'}")
    return all(os.getenv(k) for k in ("ADZUNA_APP_ID", "ADZUNA_APP_KEY"))


# ------------------------------------------------------------ 2. raw API probes
def step_adzuna_probe(role, city):
    head(f"2a. Raw Adzuna call (what='{role} intern', where='{city or 'India-wide'}')")
    params = {
        "app_id": os.getenv("ADZUNA_APP_ID"),
        "app_key": os.getenv("ADZUNA_APP_KEY"),
        "results_per_page": 20,
        "what": f"{role} intern",
        "max_days_old": jr.MAX_JOB_AGE_DAYS,
        "content-type": "application/json",
    }
    if city:
        params["where"] = city
    try:
        response = requests.get(jr.ADZUNA_URL, params=params, timeout=jr.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        return fail(f"Adzuna unreachable: {type(exc).__name__}")

    if response.status_code in (401, 403):
        return fail(f"Adzuna rejected the credentials (HTTP {response.status_code}). "
                    "Check ADZUNA_APP_ID / ADZUNA_APP_KEY in .env.")
    if response.status_code != 200:
        return fail(f"Adzuna returned HTTP {response.status_code}: {short(response.text, 160)}")
    ok("HTTP 200 from Adzuna")

    data = response.json()
    results = data.get("results", [])
    p(f"  Adzuna reports {data.get('count', '?')} total matches; {len(results)} returned.")
    if not results:
        return warn("No Adzuna results for this query/location (not necessarily a bug).")

    missing = [r.get("title") for r in results if not r.get("redirect_url")]
    (warn if missing else ok)(f"{len(results) - len(missing)}/{len(results)} raw listings have a redirect_url"
                              + (" (others are dropped by the pipeline)" if missing else ""))
    for r in results[:5]:
        loc = r.get("location") or {}
        u = urlparse(r.get("redirect_url", ""))
        p(f"   - {short(r.get('title'), 38):38} | {short((r.get('company') or {}).get('display_name'), 18):18} "
          f"| {short(loc.get('display_name'), 22):22} | area={loc.get('area')} | {u.netloc}{short(u.path, 28)}")


def step_remotive_probe(role):
    head(f"2b. Raw Remotive call (search='{role}')")
    try:
        response = requests.get(jr.REMOTIVE_URL, params={"search": role, "limit": 50},
                                timeout=jr.REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        return fail(f"Remotive unreachable/error: {type(exc).__name__}")
    ok("HTTP 200 from Remotive")
    jobs = response.json().get("jobs", [])
    p(f"  {len(jobs)} jobs returned. candidate_required_location values:")
    for loc, n in Counter(j.get("candidate_required_location", "") for j in jobs).most_common(8):
        eligible = jr.remote_eligible(loc)
        p(f"   {n:>3} x {short(loc or '(blank)', 30):30} -> {'ELIGIBLE' if eligible else 'excluded'}")
    bad = [j for j in jobs if not jr.is_job_specific_url(j.get("url", ""))]
    (warn if bad else ok)(f"{len(jobs) - len(bad)}/{len(jobs)} raw Remotive urls are job-specific"
                          + (" (generic ones are dropped by the pipeline)" if bad else ""))


# --------------------------------------------------------------- 3. link check
def probe_link(url):
    """Independent of the module's checker: returns (verdict, final_url_without_query)."""
    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=10,
                          headers={"User-Agent": "Mozilla/5.0 (InternTrust live check)"}) as r:
            final = urlparse(r.url)
            final_path = final.path.rstrip("/").lower() or "/"
            original_path = urlparse(url).path.rstrip("/").lower() or "/"
            shown = f"{final.netloc}{short(final.path, 40)}"
            if r.status_code in (404, 410):
                return f"DEAD (HTTP {r.status_code})", shown
            if final_path in jr.GENERIC_PATHS and not final.query and original_path not in jr.GENERIC_PATHS:
                return "HOMEPAGE-REDIRECT", shown
            if r.status_code < 400:
                return "OK", shown
            return f"BLOCKED/UNKNOWN (HTTP {r.status_code})", shown
    except requests.RequestException as exc:
        return f"UNREACHABLE ({type(exc).__name__})", ""


# ------------------------------------------------------------------ 4. pipeline
def show_and_check(jobs, role, location, check_links=True):
    mode = jr.location_mode(location)
    terms = jr.city_terms(location)

    p(f"\n  {'#':>2} {'TITLE':34} {'COMPANY':16} {'LOCATION':22} {'LEVEL':13} {'SRC':8} {'MATCH':>5}")
    for i, j in enumerate(jobs, 1):
        p(f"  {i:>2} {short(j['title'], 34):34} {short(j['company'], 16):16} {short(j['location'], 22):22} "
          f"{short(j.get('experience_level', '?'), 13):13} {j['source']:8} {j['match']:>5}")

    if not jobs:
        return warn("Empty result. The frontend will show 'No listings returned right now'.")

    # city first, then eligible remote
    if mode == "city":
        seen_remote = False
        for j in jobs:
            in_city = any(jr._contains_term(jr.normalize_text(j["location"]), t) for t in terms)
            if j["remote"] and not in_city:
                seen_remote = True
                if not jr.remote_eligible(j["location"]) and j["source"] == "Remotive":
                    fail(f"remote job not India/worldwide eligible: {j['title']} [{j['location']}]")
            elif not in_city:
                fail(f"job outside {location}: {j['title']} [{j['location']}]")
            elif seen_remote:
                fail(f"city job ranked after a remote job: {j['title']}")
        if not any(f.startswith(("job outside", "city job", "remote job")) for f in FAILS):
            ok(f"{location} jobs come first, followed only by eligible remote jobs")
        n_city = sum(1 for j in jobs if not j["remote"])
        p(f"  ({n_city} in {location}, {len(jobs) - n_city} remote)")
    else:
        for j in jobs:
            if j["source"] == "Remotive" and not jr.remote_eligible(j["location"]):
                fail(f"remote job not eligible: {j['title']} [{j['location']}]")

    # experience
    senior = [j["title"] for j in jobs if jr._SENIOR_TITLE.search(jr.normalize_text(j["title"]))]
    (fail if senior else ok)("no senior/lead/principal titles" if not senior else f"senior titles present: {senior}")
    levels = Counter(j.get("experience_level") for j in jobs)
    p("  Levels:", dict(levels))
    first_unspec = next((i for i, j in enumerate(jobs) if j["experience_level"] == "Not specified"), len(jobs))
    # within the same location tier, entry/intern should precede unspecified
    if levels.get("Not specified", 0) > len(jobs) / 2:
        warn("More than half are 'Not specified' (the source gave no level). Not claimed as entry-level.")

    # role relevance
    words = [w[:5] for w in jr.normalize_text(role).split() if w not in jr.ROLE_NOISE]
    off = [j["title"] for j in jobs if not any(w in jr.normalize_text(j["title"] + " " + (j.get("category") or "")) for w in words)]
    (warn if off else ok)(f"all titles relate to '{role}'" if not off else f"titles without role words: {off}")

    # links
    if check_links:
        p("\n  Apply links (checked independently, redirects followed):")
        for j in jobs:
            verdict, final = probe_link(j["link"])
            u = urlparse(j["link"])
            p(f"   [{verdict}] {short(j['title'], 30)} -> lands on {final or '-'}")
            p(f"        opens: {u.scheme}://{u.netloc}{short(u.path, 60)}  (module status: {j.get('link_status')})")
            if verdict.startswith(("HOMEPAGE", "DEAD")):
                fail(f"bad Apply link: {j['title']} -> {verdict}")
            elif not jr.is_job_specific_url(j["link"]):
                fail(f"generic Apply URL: {j['link']}")
        if not any(f.startswith("bad Apply") or f.startswith("generic") for f in FAILS):
            ok("no Apply link is dead or a homepage")
        p("  BLOCKED/UNKNOWN means the site refuses scripts; open those by hand in your browser.")


def step_pipeline(role, skills, location, top):
    head(f"3. Full pipeline: role='{role}', location='{location}'")
    out = jr.get_safe_jobs_detailed({"target_role": role, "skills": skills, "location": location}, top_n=top)
    p("  Drop counters:", out["stats"])
    show_and_check(out["jobs"], role, location)


# ----------------------------------------------------------- 5. HTTP /api/recommend
def step_http(base, email, role, skills, location):
    head(f"4. HTTP: POST {base}/api/recommend (through your running Flask app)")
    password = getpass.getpass("  Password (hidden, not stored): ")
    try:
        login = requests.post(f"{base}/api/auth/login", json={"email": email, "password": password}, timeout=20)
        token = login.json().get("token")
    except (requests.RequestException, ValueError) as exc:
        return fail(f"login request failed: {type(exc).__name__}")
    if not token:
        return fail(f"login failed (HTTP {login.status_code})")
    ok("logged in (token kept in memory only)")
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.post(f"{base}/api/recommend", headers=headers, timeout=120,
                      json={"target_role": role, "skills": skills, "location": location})
    p(f"  HTTP {r.status_code}")
    body = r.json()
    if not body.get("success"):
        return fail(f"/api/recommend error: {short(body.get('error'), 200)}")
    ok(f"success=true, count={body.get('count')}")
    needed = {"title", "company", "location", "type", "match", "link", "source", "matched_skills",
              "experience_level", "remote", "eligibility", "link_status"}
    for j in body["jobs"]:
        missing = needed - set(j)
        if missing:
            fail(f"job '{j.get('title')}' missing fields {sorted(missing)}. Is the old job_recommender.py still loaded? Restart Flask.")
            break
    else:
        ok("every job has all fields the dashboard reads")
    show_and_check(body["jobs"], role, location)

    nr = requests.post(f"{base}/api/recommend", headers=headers, json={"skills": skills}, timeout=20)
    (ok if nr.status_code == 400 and nr.json().get("needs_target_role") else fail)(
        f"missing target role -> HTTP {nr.status_code} (expected 400 + needs_target_role)")
    na = requests.post(f"{base}/api/recommend", json={}, timeout=20)
    (ok if na.status_code == 401 else fail)(f"no token -> HTTP {na.status_code} (expected 401)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="Data Scientist")
    ap.add_argument("--skills", default="Python,SQL,Machine Learning,Pandas")
    ap.add_argument("--location", action="append", help="repeatable; default: Pune, Mumbai, Worldwide")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--http", help="base URL of the running Flask app, e.g. http://localhost:5000")
    ap.add_argument("--email", help="login email for --http")
    ap.add_argument("--skip-probes", action="store_true", help="skip the raw API calls")
    args = ap.parse_args()

    skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    locations = args.location or ["Pune", "Mumbai", "Worldwide"]

    if not step_env():
        p("\nCannot call Adzuna without credentials; Remotive-only results would be misleading. Stopping.")
        return 1
    if not args.skip_probes:
        step_adzuna_probe(args.role, jr.primary_location(locations[0]) if jr.location_mode(locations[0]) == "city" else "")
        step_remotive_probe(args.role)
    for loc in locations:
        step_pipeline(args.role, skills, loc, args.top)
    if args.http:
        if not args.email:
            return fail("--http needs --email") or 1
        step_http(args.http.rstrip("/"), args.email, args.role, skills, locations[0])

    head("SUMMARY")
    if FAILS:
        p(f"{len(FAILS)} FAIL(s):")
        for f in FAILS:
            p("  -", f)
        return 1
    p("No FAILs. Now do the manual browser checks (WARN lines are worth reading).")
    return 0


if __name__ == "__main__":
    sys.exit(main())