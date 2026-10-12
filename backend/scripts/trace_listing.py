#!/usr/bin/env python
"""
Trace one listing through the recommendation pipeline (live APIs, no secrets printed).

    python scripts/trace_listing.py --watch CodeForAI --role "Data Scientist" --location Pune
    python scripts/trace_listing.py --watch codeforai --role "Data Scientist" --skills "Python,SQL"

Shows: the role/skills you passed, whether the source feed contains the listing,
whether it is India-eligible, and the stage that kept or dropped it.
Use the SAME target_role your dashboard sends: start Flask with
    $env:JOB_DEBUG="1"; python app.py
and read the first "[JOB_DEBUG ...] role=..." line when the dashboard loads.
"""
import argparse
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
try:
    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env")
except ImportError:
    pass

from services import job_recommender as jr  # noqa: E402


def matches(job, needle):
    blob = " ".join(str(job.get(k, "")) for k in ("title", "company_name", "company", "url", "link")).lower()
    return needle.lower() in blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True, help="text in the title/company/url of the listing")
    ap.add_argument("--role", default="Data Scientist")
    ap.add_argument("--skills", default="Python,SQL,Machine Learning,Pandas")
    ap.add_argument("--location", default="Pune")
    a = ap.parse_args()
    skills = [s.strip() for s in a.skills.split(",") if s.strip()]

    print(f"module version : {jr.PIPELINE_VERSION}  ({jr.__file__})")
    print(f"input          : role={a.role!r} skills={skills} location={a.location!r} "
          f"mode={jr.location_mode(a.location)}")

    print("\n[1] Raw Remotive feed (same queries the pipeline uses)")
    queries = list(dict.fromkeys([a.role, f"{a.role} intern"] + [f"{a.role} {s}" for s in skills[:2]]))
    found = None
    for q in queries:
        feed = jr.fetch_remotive(q)
        hit = next((j for j in feed if matches(j, a.watch)), None)
        print(f"    query {q!r:42} -> {len(feed):>3} jobs, watched listing: {'YES' if hit else 'no'}")
        found = found or hit
    if found:
        loc = found.get("candidate_required_location", "")
        print(f"    title    : {found.get('title')}\n    category : {found.get('category')}  type: {found.get('job_type')}")
        print(f"    location : {loc}\n    India-eligible: {jr.remote_eligible(loc)}")
        rel = jr.role_relevance(a.role, found.get("title", ""))
        print(f"    role relevance for {a.role!r}: relevant={rel[0]}  ({rel[2]})")
    else:
        print("    (not in the Remotive results for these queries)")

    print("\n[2] Final pipeline result")
    out = jr.get_safe_jobs_detailed({"target_role": a.role, "skills": skills, "location": a.location}, top_n=10)
    kept = [(n, j) for n, j in enumerate(out["jobs"], 1) if matches(j, a.watch)]
    dropped = [d for d in out["trace"]["dropped"] if matches(d, a.watch)]
    if kept:
        n, j = kept[0]
        print(f"    RETURNED at position {n}: {j['title']}\n    why: {j['relevance_reason']} | {j['eligibility']}")
        print("    -> if this is unexpected, the role you entered legitimately matches its title.")
    elif dropped:
        d = dropped[0]
        print(f"    DROPPED at stage '{d['reason']}': {d['detail']}")
    else:
        print("    not present after source fetch (not in feed, or removed by the India/worldwide eligibility filter)")
    print(f"\n    drop counters: {out['stats']}")
    print(f"    returned {len(out['jobs'])} job(s)")


if __name__ == "__main__":
    main()