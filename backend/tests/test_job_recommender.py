"""
Offline tests for services/job_recommender.py.

The sandbox/CI has no Adzuna credentials and may not reach the live APIs, so
the two API endpoints are replaced by fixtures shaped like their documented
responses. Everything else is real: requests, the link checker and its
redirect handling run against a local HTTP server.

Run:  python -m unittest tests.test_job_recommender -v     (from backend/)
      python tests/test_job_recommender.py --samples       (print sample output)
"""
import io
import json
import os
import sys
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services import job_recommender as jr  # noqa: E402

FAKE_KEY = "SECRET-KEY-123"


# ---------------------------------------------------------------- fake website
class Site(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/remote-jobs/gone", "/land/ad/redirect-home"):
            self.send_response(302)
            self.send_header("Location", "/")
        elif path in ("/remote-jobs/missing", "/land/ad/expired"):
            self.send_response(404)
        elif path == "/land/ad/blocked":
            self.send_response(403)
        else:
            self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


SERVER = HTTPServer(("127.0.0.1", 0), Site)
BASE = f"http://127.0.0.1:{SERVER.server_port}"
threading.Thread(target=SERVER.serve_forever, daemon=True).start()

NOW = "2026-10-05T09:00:00Z"


def adz(i, title, city, area, desc="", url=None, created=NOW, company="Acme"):
    return {
        "id": str(i), "title": title, "description": desc, "created": created,
        "redirect_url": url or f"{BASE}/land/ad/{i}?se=trk{i}",
        "company": {"display_name": company},
        "location": {"display_name": city, "area": area},
        "category": {"label": "IT Jobs"}, "contract_time": "full_time",
    }


PUNE = ["India", "Maharashtra", "Pune"]
ADZUNA_RESULTS = [
    adz(1001, "Data Analyst Intern", "Pune, Maharashtra", PUNE, "Python SQL internship"),
    adz(1002, "Junior Data Analyst", "Pune, Maharashtra", PUNE, "0-2 years. Python, SQL, Tableau", company="Beta"),
    adz(1003, "Data Analyst", "Pune, Maharashtra", PUNE, "Python SQL Tableau pandas dashboards", company="Gamma"),
    adz(1004, "Senior Data Analyst", "Pune, Maharashtra", PUNE, "Python SQL", company="Delta"),
    adz(1005, "Data Analyst Intern", "Mumbai, Maharashtra", ["India", "Maharashtra", "Mumbai"], "Python", company="Epsilon"),
    adz(1006, "Data Analyst Fresher", "Maharashtra", ["India", "Maharashtra"], "fresher", company="Zeta"),
    adz(1007, "Data Analyst", "Pune, Maharashtra", PUNE, "5+ years of experience in analytics", company="Eta"),
    adz(1008, "Data Analyst Trainee", "Pune, Maharashtra", PUNE, "", url=f"{BASE}/land/ad/expired", company="Theta"),
    adz(1010, "Data Analyst Intern", "Pune, Maharashtra", PUNE, "repost", url=f"{BASE}/land/ad/1010?se=other"),
    adz(1011, "Data Analyst - Work from Home", "India", ["India"], "Remote role, Python", company="Iota"),
    adz(1012, "Data Analyst Intern", "Pune, Maharashtra", PUNE, "", url=f"{BASE}/land/ad/redirect-home", company="Kappa"),
    adz(1013, "Marketing Intern", "Pune, Maharashtra", PUNE, "", company="Lambda"),
    adz(1014, "Data Analyst", "Pune, Maharashtra", PUNE, "SQL", url=f"{BASE}/land/ad/blocked", company="Mu"),
    adz(1015, "Data Analyst Intern", "Pune, Maharashtra", PUNE, "", created="2026-01-01T00:00:00Z", company="Nu"),
    adz(1016, "Data Analyst Intern", "Pune, Maharashtra", PUNE, "", url=f"{BASE}/", company="Xi"),
    adz(2001, "Data Analyst Intern", "Mumbai, Maharashtra", ["India", "Maharashtra", "Mumbai"], "Python SQL", company="MumbaiCo"),
]


def rem(i, title, loc, desc="", jtype="full_time", slug=None, company="RemCo"):
    return {
        "id": i, "title": title, "company_name": company, "description": desc,
        "job_type": jtype, "category": "Data", "candidate_required_location": loc,
        "publication_date": "2026-10-04T10:00:00",
        "url": f"{BASE}/remote-jobs/data/{slug or 'job-' + str(i)}",
    }


# Real listing the user reported (fetched from remotive.com on 2026-10-10). Title, company, URL,
# category, contract type and location are exact; the description is a short summary written
# for this test (not a copy), and deliberately mentions Python / SQL / AI / data structures.
CODEFORAI_URL = ("https://remotive.com/remote-jobs/software-development/"
                 "software-engineer-ai-code-trainer-python-web-full-stack-2091149")
CODEFORAI = {
    "id": 2091149,
    "title": "Software Engineer / AI Code Trainer (Python, Web, Full Stack)",
    "company_name": "CodeForAI",
    "url": CODEFORAI_URL,
    "category": "Software Development",
    "job_type": "contract",
    "candidate_required_location":
        "USA, UK, India, Australia, Ireland, New Zealand, Philippines, Pakistan, Singapore, Mexico",
    "publication_date": "2026-10-05T00:00:00",
    "tags": ["python", "ai", "full stack", "machine learning", "data analysis", "sql"],
    "description": ("Remote contract. Evaluate AI-generated code and write reference solutions. "
                    "Software engineers, full-stack and backend developers; Python, JavaScript, Java; "
                    "algorithms, data structures, SQL databases, pandas, machine learning, data science "
                    "and data analysis experience welcome. Train LLMs on code."),
}
# every skill a Data Scientist student might list: the listing text matches ALL of them
ALL_SKILLS = ("Python", "SQL", "Machine Learning", "Pandas", "AI", "Data Science", "Data Analysis")

REMOTIVE_JOBS = [
    rem(1, "Data Analyst Intern", "Worldwide", "Python SQL", "internship", company="GlobalCo"),
    rem(2, "Data Analyst", "USA Only", "SQL"),
    rem(3, "Data Analyst (Python)", "Worldwide", "4+ years of commercial experience", company="Lemon.io"),
    rem(4, "Data Analyst", "India", "SQL", slug="../gone", company="IndiaCo"),
    rem(5, "Junior Data Analyst", "", "SQL"),
    rem(6, "Data Analyst Intern", "Worldwide (not India)", ""),
    rem(7, "Data Analyst", "Anywhere in the USA", ""),
    rem(8, "Data Analyst Trainee", "APAC", ""),
    {**rem(9, "Data Analyst Intern", "Worldwide", ""), "url": "https://remotive.com/"},
    rem(10, "Data Analyst", "Worldwide", "SQL", slug="../missing"),
    rem(11, "Junior Data Analyst", "India", "entry level, SQL pandas", company="IndiaJr"),
    CODEFORAI,
]
# the "../" slugs above resolve on the fake site to /remote-jobs/gone|missing
for j in REMOTIVE_JOBS:
    j["url"] = j["url"].replace("/data/../", "/")

API_CALLS = []
LINK_REQUESTS = []
_real_get = requests.get


def fake_get(url, params=None, **kw):
    if url == jr.ADZUNA_URL:
        API_CALLS.append(("adzuna", dict(params)))
        resp = requests.Response()
        resp.status_code = 200
        resp._content = json.dumps({"results": ADZUNA_RESULTS}).encode()
        return resp
    if url == jr.REMOTIVE_URL:
        API_CALLS.append(("remotive", dict(params)))
        resp = requests.Response()
        resp.status_code = 200
        resp._content = json.dumps({"jobs": REMOTIVE_JOBS}).encode()
        return resp
    if url.startswith("https://remotive.com/remote-jobs/"):
        LINK_REQUESTS.append(url)
        resp = requests.Response()
        resp.status_code = 200
        resp.url = url
        resp.raw = io.BytesIO(b"ok")
        resp._content = b"ok"
        return resp
    return _real_get(url, params=params, **kw)


def run(role="Data Analyst", skills=("Python", "SQL"), location="Pune", **extra):
    jr._CACHE.clear()
    API_CALLS.clear()
    LINK_REQUESTS.clear()
    env = {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": FAKE_KEY}
    with mock.patch.dict(os.environ, env), \
         mock.patch.object(requests, "get", fake_get), \
         redirect_stdout(io.StringIO()):
        return jr.get_safe_jobs_detailed(
            {"target_role": role, "skills": list(skills), "location": location, **extra},
            top_n=20,
        )


class Tests(unittest.TestCase):
    # -- 1. Adzuna really called, credentials come from env, results combined
    def test_adzuna_called_and_combined(self):
        out = run(location="Pune")
        calls = [c for c in API_CALLS if c[0] == "adzuna"]
        self.assertGreaterEqual(len(calls), 3)
        self.assertTrue(all(c[1]["where"] == "Pune" for c in calls))
        self.assertTrue(all(c[1]["app_id"] == "id123" for c in calls))
        self.assertIn("remotive", {c[0] for c in API_CALLS})
        self.assertEqual({j["source"] for j in out["jobs"]}, {"Adzuna", "Remotive"})

    def test_missing_credentials_no_adzuna_call(self):
        jr._CACHE.clear(); API_CALLS.clear()
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(requests, "get", fake_get), \
             redirect_stdout(io.StringIO()) as buf:
            jr.get_safe_jobs({"target_role": "Data Analyst", "location": "Pune"})
        self.assertNotIn("adzuna", {c[0] for c in API_CALLS})
        self.assertIn("credentials are missing", buf.getvalue())

    # -- 2. URLs preserved exactly, never homepage
    def test_urls_preserved_and_job_specific(self):
        out = run()
        by_id = {j["link"] for j in out["jobs"]}
        self.assertIn(f"{BASE}/land/ad/1001?se=trk1001", by_id)          # Adzuna redirect_url intact
        self.assertIn(f"{BASE}/remote-jobs/data/job-1", by_id)           # Remotive url intact
        for j in out["jobs"]:
            self.assertNotIn(j["link"].rstrip("/"), (BASE, "https://remotive.com"))
            self.assertTrue(jr.is_job_specific_url(j["link"]))

    def test_dead_and_homepage_links_dropped_blocked_kept_unverified(self):
        out = run()
        links = {j["link"]: j for j in out["jobs"]}
        self.assertFalse(any("/expired" in l or "redirect-home" in l or l.endswith("/gone")
                             or l.endswith("/missing") for l in links))
        blocked = links[f"{BASE}/land/ad/blocked"]
        self.assertEqual(blocked["link_status"], "unverified")
        self.assertEqual(links[f"{BASE}/land/ad/1001?se=trk1001"]["link_status"], "verified")
        self.assertGreaterEqual(out["stats"]["dead_link"], 4)
        self.assertGreaterEqual(out["stats"]["invalid_link"], 2)

    # -- 3. Location
    def test_pune_only_pune_or_eligible_remote(self):
        out = run(location="Pune")
        for j in out["jobs"]:
            if j["source"] == "Adzuna" and not j["remote"]:
                self.assertIn("Pune", j["location"])
            else:
                self.assertTrue(j["remote"])
        titles_locs = {(j["company"], j["location"]) for j in out["jobs"]}
        self.assertFalse(any("Mumbai" in l for _, l in titles_locs))
        self.assertFalse(any(l == "Maharashtra" for _, l in titles_locs))

    def test_pune_ranked_before_remote(self):
        jobs = run(location="Pune")["jobs"]
        kinds = ["pune" if (j["source"] == "Adzuna" and not j["remote"]) else "remote" for j in jobs]
        self.assertEqual(kinds, sorted(kinds, key=lambda k: k != "pune"))
        self.assertEqual(kinds[0], "pune")

    def test_mumbai_selection_returns_mumbai_first(self):
        jobs = run(location="Mumbai")["jobs"]
        self.assertIn("Mumbai", jobs[0]["location"])
        self.assertFalse(any("Pune" in j["location"] for j in jobs))

    def test_free_text_profile_location(self):
        out = run(location="Pune, Maharashtra")
        calls = [c for c in API_CALLS if c[0] == "adzuna"]
        self.assertTrue(all(c[1]["where"] == "Pune" for c in calls))
        self.assertIn("Pune", out["jobs"][0]["location"])
        self.assertEqual(jr.location_mode("Worldwide / Remote"), "global")
        self.assertEqual(jr.location_mode("Anywhere in India"), "india")

    # -- 4. Remote eligibility
    def test_remote_only_india_or_worldwide(self):
        for loc in ("Pune", "Worldwide", "India"):
            for j in run(location=loc)["jobs"]:
                if j["source"] == "Remotive":
                    self.assertTrue(jr.remote_eligible(j["location"]), j)
        for bad in ("USA Only", "", "APAC", "Worldwide (not India)", "Anywhere in the USA", "Europe"):
            self.assertFalse(jr.remote_eligible(bad), bad)
        for good in ("India", "Worldwide", "Anywhere", "Anywhere in India", "Asia, India"):
            self.assertTrue(jr.remote_eligible(good), good)

    def test_unlisted_city_has_only_eligible_remote(self):
        jobs = run(location="Jaipur")["jobs"]
        self.assertTrue(jobs)
        self.assertTrue(all(j["remote"] for j in jobs))

    # -- 5. Experience
    def test_senior_excluded_and_entry_first(self):
        out = run(location="Pune")
        names = [j["title"] for j in out["jobs"]]
        self.assertNotIn("Senior Data Analyst", names)
        self.assertFalse(any(j["company"] == "Lemon.io" for j in out["jobs"]))
        pune = [j for j in out["jobs"] if not j["remote"]]
        levels = [j["experience_level"] for j in pune]
        first_unspec = levels.index("Not specified")
        self.assertTrue(all(l in ("Internship", "Entry Level") for l in levels[:first_unspec]))
        self.assertTrue(all(l == "Not specified" for l in levels[first_unspec:]))

    def test_eligibility_beats_skill_score(self):
        pune = [j for j in run(location="Pune")["jobs"] if not j["remote"]]
        gamma = next(j for j in pune if j["company"] == "Gamma")      # 100% skill match, no level
        junior = next(j for j in pune if j["company"] == "Beta")
        self.assertGreater(pune.index(gamma), pune.index(junior))

    def test_senior_allowed_only_when_requested(self):
        self.assertTrue(any("Senior" in j["title"] for j in run(role="Senior Data Analyst")["jobs"]))
        self.assertFalse(any("Senior" in j["title"] for j in run()["jobs"]))

    def test_classifier_edge_cases(self):
        c = lambda t, d="", jt="": jr.classify_experience({"title": t, "description": d, "job_type": jt})["level"]
        self.assertEqual(c("Lead Generation Intern"), "internship")
        self.assertEqual(c("Product Manager Intern"), "internship")
        self.assertEqual(c("Team Lead"), "senior")
        self.assertEqual(c("Data Analyst", "3-5 years experience"), "senior")
        self.assertEqual(c("Data Analyst", "we were founded 10 years ago"), "unspecified")
        self.assertEqual(c("Data Analyst", "", "internship"), "internship")
        self.assertEqual(c("Graduate Engineer Trainee"), "entry")
        self.assertEqual(c("Management Trainee"), "entry")

    # -- 6. Dupes, empty state, secrets
    def test_no_duplicates(self):
        jobs = run(location="Pune")["jobs"]
        links = [j["link"] for j in jobs]
        self.assertEqual(len(links), len(set(links)))
        fp = [(j["title"], j["company"], j["location"]) for j in jobs]
        self.assertEqual(len(fp), len(set(fp)))

    def test_empty_state_for_unmatched_role(self):
        self.assertEqual(run(role="Astronaut")["jobs"], [])

    def test_api_key_never_logged(self):
        def boom(url, params=None, **kw):
            if url == jr.ADZUNA_URL:
                resp = requests.Response(); resp.status_code = 401
                raise requests.HTTPError(f"401 for url: {url}?app_key={FAKE_KEY}", response=resp)
            return fake_get(url, params, **kw)
        jr._CACHE.clear()
        with mock.patch.dict(os.environ, {"ADZUNA_APP_ID": "x", "ADZUNA_APP_KEY": FAKE_KEY}), \
             mock.patch.object(requests, "get", boom), redirect_stdout(io.StringIO()) as buf:
            jr.get_safe_jobs({"target_role": "Data Analyst", "location": "Pune"})
        self.assertNotIn(FAKE_KEY, buf.getvalue())
        self.assertIn("HTTP 401", buf.getvalue())


class CodeForAIRegression(unittest.TestCase):
    """The listing that kept appearing for Data Scientist students."""

    def _found(self, out):
        return [j for j in out["jobs"] if j["link"] == CODEFORAI_URL or j["company"] == "CodeForAI"]

    def test_excluded_for_data_science_and_analyst_roles_in_every_location(self):
        roles = ["Data Scientist", "Data Science", "Data Science Intern", "Data Analyst",
                 "Machine Learning Engineer", "AI/ML Engineer", "Data Scientist / ML Engineer",
                 "Business Intelligence Analyst", "Python Data Scientist"]
        for role in roles:
            for location in ("Pune", "Mumbai", "Worldwide", "India"):
                with self.subTest(role=role, location=location):
                    out = run(role=role, skills=ALL_SKILLS, location=location)
                    self.assertEqual(self._found(out), [])
                    self.assertTrue(any(d["link"] == CODEFORAI_URL and d["reason"] == "irrelevant_role"
                                        for d in out["trace"]["dropped"]),
                                    "listing must be dropped at the relevance stage, with a recorded reason")

    def test_perfect_skill_match_cannot_override_unrelated_title(self):
        jobs = run(role="Data Scientist", skills=ALL_SKILLS, location="Worldwide")["jobs"]
        self.assertFalse(any("Software Engineer" in j["title"] for j in jobs))
        self.assertTrue(all(j["relevance_reason"] for j in jobs))

    def test_allowed_when_user_asks_for_matching_software_role(self):
        for role in ("Software Engineer", "Software Developer", "AI Code Trainer"):
            with self.subTest(role=role):
                out = run(role=role, skills=("Python",), location="Worldwide")
                hit = self._found(out)
                self.assertEqual(len(hit), 1)
                self.assertEqual(hit[0]["link"], CODEFORAI_URL)          # URL untouched
                self.assertEqual(hit[0]["source"], "Remotive")
                self.assertTrue(hit[0]["remote"])
                self.assertIn("India", hit[0]["eligibility"])            # eligibility is in the data
                self.assertEqual(hit[0]["experience_level"], "Not specified")

    def test_allowed_for_software_role_in_pune_after_local_jobs(self):
        out = run(role="Software Engineer", skills=("Python",), location="Pune")
        self.assertEqual([j["company"] for j in out["jobs"]], ["CodeForAI"])
        self.assertTrue(out["jobs"][0]["remote"])

    def test_loose_roles_that_only_share_a_word_do_not_match(self):
        # old rule: all words anywhere in title / 50% title + description rescue
        for role in ("AI Engineer", "Python Developer", "Web Analyst", "Machine Learning Intern"):
            with self.subTest(role=role):
                self.assertEqual(self._found(run(role=role, skills=ALL_SKILLS, location="Worldwide")), [])

    def test_relevance_runs_before_link_validation(self):
        run(role="Data Scientist", skills=ALL_SKILLS, location="Worldwide")
        self.assertNotIn(CODEFORAI_URL, LINK_REQUESTS)        # irrelevant jobs are never link-checked
        run(role="Software Engineer", skills=(), location="Worldwide")
        self.assertIn(CODEFORAI_URL, LINK_REQUESTS)           # relevant ones are, separately

    def test_no_relevant_jobs_gives_empty_list_not_padding(self):
        jr._CACHE.clear()
        with mock.patch.object(sys.modules[__name__], "REMOTIVE_JOBS", [CODEFORAI]), \
             mock.patch.object(sys.modules[__name__], "ADZUNA_RESULTS", []):
            out = run(role="Data Scientist", skills=ALL_SKILLS, location="Pune")
        self.assertEqual(out["jobs"], [])
        self.assertEqual(out["stats"]["irrelevant_role"], 1)

    def test_debug_trace_shows_input_and_drop_reason_without_secrets(self):
        jr._CACHE.clear()
        env = {"ADZUNA_APP_ID": "id123", "ADZUNA_APP_KEY": FAKE_KEY, "JOB_DEBUG": "1"}
        with mock.patch.dict(os.environ, env), mock.patch.object(requests, "get", fake_get), \
             redirect_stdout(io.StringIO()) as buf:
            jr.get_safe_jobs({"target_role": "Data Scientist", "skills": ["Python"], "location": "Pune"})
        text = buf.getvalue()
        self.assertIn("role='Data Scientist'", text)
        self.assertIn("DROP irrelevant_role", text)
        self.assertIn("Software Engineer / AI Code Trainer", text)
        self.assertNotIn(FAKE_KEY, text)


class RelevanceRules(unittest.TestCase):
    def rel(self, role, title):
        return jr.role_relevance(role, title)[0]

    def test_data_scientist_family(self):
        for t in ("Data Scientist", "Data Science Intern", "Machine Learning Engineer", "ML Intern",
                  "AI/ML Intern", "NLP Research Intern", "Data Analyst Trainee", "Analytics Intern",
                  "Junior Data Scientist", "AI Intern"):
            self.assertTrue(self.rel("Data Scientist", t), t)
        for t in ("Software Engineer", "Full Stack Developer", "AI Code Trainer", "Backend Developer (Python)",
                  "Software Engineer / AI Code Trainer (Python, Web, Full Stack)", "Data Engineer",
                  "Marketing Intern", "Frontend Developer", "AI Trainer"):
            self.assertFalse(self.rel("Data Scientist", t), t)

    def test_data_analyst_family(self):
        for t in ("Data Analyst", "Business Intelligence Analyst", "BI Developer", "SQL Developer Intern",
                  "Power BI Intern", "Business Analyst Trainee", "Analytics Intern", "Reporting Analyst"):
            self.assertTrue(self.rel("Data Analyst", t), t)
        for t in ("Software Engineer", "Full Stack Developer", "AI Code Trainer", "Data Scientist",
                  "Machine Learning Engineer", "Web Developer"):
            self.assertFalse(self.rel("Data Analyst", t), t)

    def test_unknown_roles_need_the_phrase_in_order(self):
        self.assertTrue(self.rel("Software Engineer", "Software Engineer / AI Code Trainer"))
        self.assertTrue(self.rel("Software Developer", "Senior Software Engineer"))
        self.assertTrue(self.rel("Marketing Intern", "Digital Marketing Intern"))
        self.assertFalse(self.rel("AI Engineer", "Software Engineer / AI Code Trainer"))
        self.assertFalse(self.rel("Engineer Software", "Software Engineer"))


def print_samples():
    for loc in ("Pune", "Worldwide"):
        out = run(location=loc)
        print(f"\n=== location={loc}  stats={out['stats']}")
        for j in out["jobs"]:
            print(f"- {j['title']} | {j['company']} | {j['location']} | {j['experience_level']} | "
                  f"{j['source']} | match={j['match']} | {j['link_status']}\n    {j['link']}")


if __name__ == "__main__":
    if "--samples" in sys.argv:
        print_samples()
    else:
        unittest.main(verbosity=2)