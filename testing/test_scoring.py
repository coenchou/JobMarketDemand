"""
Scoring regression tests.

Every calibration constant in this project was fitted by eye against a handful
of sample resumes. That is fine as a method and fatal as a safety net: nothing
stopped a change from silently moving every score by fifteen points. These
tests pin the behaviour that matters.

They are deliberately offline and deterministic — fixed skill lists rather than
parsed resumes, so neither the LLM extractor nor the network is in the loop.
Run with:

    PYTHONPATH=. python -m unittest discover -s testing
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.career_stage import career_stage, score_track_record, stage_weights
from src.field_direction import infer_field
from src.student_profile import (
    COLLEGE,
    HIGH_SCHOOL,
    detect_student,
    score_application,
)
from src.gap_engine import compute_skill_gaps
from src.pipeline import _pick_default_role, _rank_roles
from src.naming import pretty_skill
from src.skill_difficulty import (
    _difficulty_from_table,
    batch_difficulty_months,
    get_difficulty_months,
)
from src.skill_implication import classify_occupation_tools, gap_priority
from src.skill_matcher import match_skill_to_tools_strict, score_soc_candidates
from src.skill_score import compute_skill_capital
from src.simulator import simulate_additions
from src.target_role import extract_posting_skills, occupation_frame

ML_ENGINEER = [
    "Python", "Scala", "SQL", "bash", "TensorFlow", "PyTorch", "scikit-learn",
    "XGBoost", "HuggingFace Transformers", "MLflow", "Kubeflow", "AWS SageMaker",
    "Docker", "Kubernetes", "Airflow", "Apache Spark", "Kafka", "dbt",
    "PostgreSQL", "Redshift", "BigQuery", "NLP", "A/B testing",
    "feature engineering", "system design", "team leadership", "Git", "GitHub",
    "statistical analysis",
]
DATA_ANALYST = [
    "Python", "SQL", "Excel", "Tableau", "pandas", "matplotlib",
    "data cleaning", "PostgreSQL", "Google BigQuery", "A/B testing",
]
JUNIOR = ["Excel", "PowerPoint", "Word"]

DATA_SCIENTIST_SOC = "15-2051"


class StrictMatchingTests(unittest.TestCase):
    """Crediting a candidate with a tool demands precision, not recall."""

    def test_exact_and_phrase_names_match(self):
        self.assertIn("apache spark", match_skill_to_tools_strict("Apache Spark"))
        self.assertIn("microsoft power bi", match_skill_to_tools_strict("Power BI"))

    def test_shared_vendor_word_does_not_credit_siblings(self):
        hits = match_skill_to_tools_strict("Apache Flink")
        self.assertNotIn("apache hadoop", hits)
        self.assertNotIn("apache cassandra", hits)
        self.assertNotIn("apache pig", hits)

    def test_generic_phrase_claims_nothing(self):
        self.assertEqual(match_skill_to_tools_strict("system design"), set())
        self.assertEqual(match_skill_to_tools_strict("feature engineering"), set())

    def test_single_letter_skill_does_not_substring_match(self):
        hits = match_skill_to_tools_strict("R")
        self.assertTrue(all(h == "r" for h in hits), hits)


class ImplicationTests(unittest.TestCase):
    """Missing != absent: entailed skills count, commodity skills don't matter."""

    @classmethod
    def setUpClass(cls):
        tools = ["Python", "Linux", "Microsoft Excel", "Microsoft PowerPoint",
                 "GitLab", "IBM Terraform", "MySQL", "Apache Hadoop"]
        cls.status = classify_occupation_tools(tools, ML_ENGINEER)

    def test_prerequisite_of_an_advanced_skill_is_implied(self):
        self.assertIn(self.status["Python"]["status"], ("held", "implied"))
        self.assertEqual(self.status["Linux"]["status"], "implied")

    def test_substitute_is_implied(self):
        self.assertEqual(self.status["GitLab"]["status"], "implied")
        self.assertEqual(self.status["MySQL"]["status"], "implied")

    def test_commodity_skills_are_neutral(self):
        self.assertEqual(self.status["Microsoft Excel"]["status"], "commodity")
        self.assertEqual(self.status["Microsoft PowerPoint"]["status"], "commodity")

    def test_real_gaps_survive(self):
        self.assertEqual(self.status["IBM Terraform"]["status"], "gap")
        self.assertEqual(self.status["Apache Hadoop"]["status"], "gap")

    def test_beginner_gets_no_commodity_forgiveness(self):
        status = classify_occupation_tools(["Microsoft Word"], ["Microsoft Excel"])
        self.assertNotEqual(status["Microsoft Word"]["status"], "commodity")

    def test_priority_rises_with_demand_and_relevance(self):
        self.assertGreater(gap_priority(3.0, 0.8), gap_priority(0.0, 0.8))
        self.assertGreater(gap_priority(3.0, 0.8), gap_priority(3.0, 0.1))
        self.assertGreater(gap_priority(0.0, 0.5, market_share=0.4),
                           gap_priority(0.0, 0.5, market_share=0.0))


class DifficultyTests(unittest.TestCase):
    def test_curated_values(self):
        self.assertEqual(get_difficulty_months("Python"), 3.0)
        self.assertEqual(get_difficulty_months("Microsoft Word"), 0.3)

    def test_token_subset_beats_substring(self):
        self.assertIsNone(_difficulty_from_table("qorbulator"))
        self.assertEqual(get_difficulty_months("Apache Spark"), 5.0)

    def test_batch_matches_single(self):
        names = ["Python", "Kubernetes", "Nonexistent Widget 9000"]
        self.assertEqual(batch_difficulty_months(names),
                         [get_difficulty_months(n) for n in names])


class SkillCapitalTests(unittest.TestCase):
    """The headline component, pinned against hand-checked baselines."""

    BASELINES = {
        "ml_engineer": (ML_ENGINEER, 0.763, 0.05),
        "data_analyst": (DATA_ANALYST, 0.439, 0.05),
        "junior": (JUNIOR, 0.403, 0.05),
    }

    def test_scores_stay_within_tolerance(self):
        for name, (skills, expected, tol) in self.BASELINES.items():
            with self.subTest(profile=name):
                cap = compute_skill_capital(skills, DATA_SCIENTIST_SOC)
                self.assertAlmostEqual(
                    cap["score"], expected, delta=tol,
                    msg=f"{name}: {cap['score']} drifted from {expected}")

    def test_stronger_profile_scores_higher(self):
        strong = compute_skill_capital(ML_ENGINEER, DATA_SCIENTIST_SOC)["score"]
        weak = compute_skill_capital(JUNIOR, DATA_SCIENTIST_SOC)["score"]
        self.assertGreater(strong, weak)

    def test_thin_resume_is_pulled_toward_the_prior(self):
        cap = compute_skill_capital(JUNIOR, DATA_SCIENTIST_SOC)
        self.assertLess(cap["evidence"], 0.5)
        self.assertGreater(cap["score"], cap["measured"])

    def test_focus_list_is_short_and_actionable(self):
        cap = compute_skill_capital(ML_ENGINEER, DATA_SCIENTIST_SOC)
        self.assertLessEqual(len(cap["focus_skills"]), 5)
        self.assertTrue(all(g["skill"] for g in cap["focus_skills"]))

    def test_commodity_tools_leave_the_denominator(self):
        cap = compute_skill_capital(ML_ENGINEER, DATA_SCIENTIST_SOC)
        self.assertGreater(cap["ignored_basic"], 0)


class GapEngineTests(unittest.TestCase):
    def test_gaps_exclude_implied_and_commodity(self):
        result = compute_skill_gaps(ML_ENGINEER, DATA_SCIENTIST_SOC)
        gap_names = {g["skill"].lower() for g in result["gaps"]}
        self.assertNotIn("microsoft excel", gap_names)
        self.assertNotIn("gitlab", gap_names)

    def test_strengths_flag_inferred_entries(self):
        result = compute_skill_gaps(ML_ENGINEER, DATA_SCIENTIST_SOC)
        self.assertTrue(any(s["implied"] for s in result["strengths"]))
        self.assertTrue(all("skill" in s for s in result["strengths"]))

    def test_coverage_is_a_fraction(self):
        result = compute_skill_gaps(DATA_ANALYST, DATA_SCIENTIST_SOC)
        self.assertGreaterEqual(result["coverage"], 0.0)
        self.assertLessEqual(result["coverage"], 1.0)


class PostingTargetTests(unittest.TestCase):
    JD = """Senior Data Engineer
    Build data pipelines with Python, dbt and Apache Airflow. Analyze usage in
    Snowflake. Experience with Kubernetes, Terraform and AWS required.
    Familiarity with Spark and Kafka is a plus. Strong SQL essential.
    You will go to market with product teams and access sensitive systems."""

    def test_extracts_real_requirements(self):
        found = {name for name, _ in extract_posting_skills(self.JD)}
        for expected in ("Python", "Snowflake", "Kubernetes", "Terraform", "SQL"):
            self.assertIn(expected, found)

    def test_ignores_prose_that_looks_like_tools(self):
        found = {name.lower() for name, _ in extract_posting_skills(self.JD)}
        for prose in ("go", "access", "analyze", "email"):
            self.assertNotIn(prose, found)

    def test_posting_can_be_scored_like_an_occupation(self):
        from src.target_role import posting_frame
        cap = compute_skill_capital(ML_ENGINEER, tools=posting_frame(self.JD))
        self.assertGreater(cap["score"], 0.5)
        self.assertGreater(cap["matched_count"], 3)


class SimulatorTests(unittest.TestCase):
    def test_learning_a_required_skill_raises_the_score(self):
        tools = occupation_frame(DATA_SCIENTIST_SOC)
        result = simulate_additions(
            DATA_ANALYST, ["PyTorch"], tools=tools, exp_fit=0.8, edu_fit=1.0,
            weights=stage_weights("early"))
        entry = result["skills"][0]
        self.assertEqual(entry["skill"], "PyTorch")
        self.assertGreater(entry["delta"], 0)

    def test_irrelevant_skill_reports_no_movement(self):
        tools = occupation_frame(DATA_SCIENTIST_SOC)
        result = simulate_additions(
            DATA_ANALYST, ["Underwater Basket Weaving"],
            tools=tools, exp_fit=0.8, edu_fit=1.0, weights=stage_weights("early"))
        entry = result["skills"][0]
        self.assertEqual(entry["delta"], 0.0)
        self.assertFalse(entry["measurable"])


class CareerStageTests(unittest.TestCase):
    """Weights have to move with the profile, not sit fixed for everyone."""

    def test_stage_boundaries(self):
        self.assertEqual(career_stage(None), "entry")
        self.assertEqual(career_stage(0), "entry")
        self.assertEqual(career_stage(3), "early")
        self.assertEqual(career_stage(7), "mid")
        self.assertEqual(career_stage(15), "senior")

    def test_weights_always_sum_to_one(self):
        for stage in ("entry", "early", "mid", "senior"):
            self.assertAlmostEqual(sum(stage_weights(stage).values()), 1.0, places=6)

    def test_education_decays_and_experience_grows_with_seniority(self):
        entry, senior = stage_weights("entry"), stage_weights("senior")
        self.assertGreater(entry["education"], senior["education"])
        self.assertLess(entry["experience"], senior["experience"])
        self.assertLess(entry["track_record"], senior["track_record"])

    def test_graduate_is_not_scored_mainly_on_missing_experience(self):
        w = stage_weights(career_stage(None))
        self.assertGreater(w["education"], w["experience"])

    def test_track_record_reads_achievements(self):
        senior = score_track_record([
            "Led a team of 12 engineers through a platform migration",
            "Scaled the recommendation platform to 8M users, cutting latency 40%",
            "Published in RecSys Workshop and maintains an open-source library",
            "Architected and shipped the end-to-end model serving stack",
            "Reduced infrastructure spend by $400k across two production systems",
            "Mentored six engineers and set the technical direction for the team",
        ])
        thin = score_track_record(["Bachelor's degree"])
        self.assertGreater(senior["raw"], 0.8)
        self.assertGreater(senior["score"], thin["score"])
        self.assertIn("led people or set direction", senior["signals"])

    def test_thin_evidence_is_pulled_toward_the_prior(self):
        headers = score_track_record(
            ["Software Engineer at Apple Inc."],
            ["Senior Software Engineer 05/03/2023 - 01/26/2024", "Apple Inc. Cupertino, CA"])
        self.assertLess(headers["evidence"], 0.5)
        self.assertGreater(headers["score"], headers["raw"])

    def test_track_record_handles_an_empty_resume(self):
        empty = score_track_record([], [])
        self.assertEqual(empty["evidence"], 0.0)
        self.assertEqual(empty["score"], 0.5)
        self.assertTrue(empty["missing"])


class RoleRankingTests(unittest.TestCase):
    """The role that leads should be the one they are most hirable for."""

    CANDIDATES = [
        {"soc_code": "15-1254", "title": "Web Developers",
         "match_score": 6.8, "blended_score": 0.99},
        {"soc_code": "15-1252", "title": "Software Developers",
         "match_score": 6.1, "blended_score": 0.71},
        {"soc_code": "25-2023", "title": "Career/Technical Education Teachers",
         "match_score": 6.4, "blended_score": 0.51},
    ]

    def setUp(self):
        self.ranked = _rank_roles(
            self.CANDIDATES, ML_ENGINEER, 5, "Bachelor's",
            {"score": 1.0, "signals": []})

    def test_every_shortlisted_role_is_scored(self):
        self.assertEqual(len(self.ranked), 3)
        self.assertTrue(all(r["hirability"] > 0 for r in self.ranked))

    def test_poor_fit_cannot_lead_on_a_low_entry_bar(self):
        teacher = next(r for r in self.ranked if r["soc_code"] == "25-2023")
        self.assertFalse(teacher["eligible"])
        self.assertNotEqual(_pick_default_role(self.ranked), "25-2023")

    def test_listed_closest_first(self):
        fits = [r["fit"] for r in self.ranked]
        self.assertEqual(fits, sorted(fits, reverse=True))

    def test_leader_is_the_most_hirable_eligible_role(self):
        eligible = [r for r in self.ranked if r["eligible"]]
        self.assertEqual(_pick_default_role(self.ranked),
                         max(eligible, key=lambda r: r["hirability"])["soc_code"])


HS_RESUME = """ALEX RIVERA
Lincoln High School, Class of 2027
EDUCATION
Lincoln High School - Expected graduation June 2027
GPA: 3.9 unweighted. Relevant coursework: AP Computer Science A, AP Calculus BC
SAT: 1480
EXPERIENCE
Founder & President - Lincoln Coding Club, Sept 2024 - Present
Grew club from 6 to 45 members; organized a hackathon with 120 participants
Software Development Intern - LocalBiz Solutions, summer 2025
Built an inventory tool in Python and SQLite used by 3 storefronts
AWARDS
2nd place, State Science Fair. USACO Silver Division.
SKILLS
Python, Java, JavaScript, React, SQL, Git, Firebase, pandas
"""

PRO_RESUME = """DANA LEE
EXPERIENCE
Senior Software Engineer - Acme Corp, 2019 - Present
Led a team of 8; scaled the platform to 4M users, cutting latency 40%
EDUCATION
B.S. Computer Science, State University, graduated 2015
SKILLS
Python, Kubernetes, Docker, PostgreSQL, Terraform, React
"""


class StudentDetectionTests(unittest.TestCase):
    """A student is not a job candidate and must not be scored as one."""

    def test_high_school_is_detected(self):
        d = detect_student(HS_RESUME, education_level=None)
        self.assertIsNotNone(d)
        self.assertEqual(d["kind"], HIGH_SCHOOL)
        self.assertEqual(d["grad_year"], 2027)

    def test_college_is_detected(self):
        text = ("B.S. Electrical Engineering, UC Berkeley. Expected graduation "
                "May 2027. Undergraduate researcher. Coursework: Algorithms.")
        d = detect_student(text, education_level="Bachelor's")
        self.assertIsNotNone(d)
        self.assertEqual(d["kind"], COLLEGE)

    def test_working_professional_is_not_a_student(self):
        self.assertIsNone(detect_student(PRO_RESUME, education_level="Bachelor's"))

    def test_every_sample_resume_is_classified_correctly(self):
        """Fixtures named student_* must be detected; the rest must not be."""
        import glob
        from src.parser import parse_resume
        for path in sorted(glob.glob("resumes/*.txt")):
            with self.subTest(resume=path):
                parsed = parse_resume(path)
                got = detect_student(
                    Path(path).read_text(), parsed.get("education_lines"),
                    parsed.get("education_level"))
                expected_student = Path(path).name.startswith("student_")
                self.assertEqual(bool(got), expected_student)

    def test_ordinary_words_do_not_trigger_detection(self):
        text = ("Directed a play and act as the primary contact. 2019 - Present. "
                "Managed university vendor relationships.")
        self.assertIsNone(detect_student(text, education_level="Master's"))


class StudentScoringTests(unittest.TestCase):
    def test_dimensions_read_the_resume(self):
        r = score_application(HS_RESUME, kind=HIGH_SCHOOL)
        found = {d["key"]: d["found"] for d in r["dimensions"]}
        self.assertTrue(found["recognition"], "science fair and USACO were missed")
        self.assertTrue(found["initiative"], "founding a club was missed")
        self.assertTrue(found["output"], "the internship was missed")

    def test_strong_profile_beats_a_bare_one(self):
        bare = score_application("Lincoln High School. Class of 2027.", kind=HIGH_SCHOOL)
        strong = score_application(HS_RESUME, kind=HIGH_SCHOOL)
        self.assertGreater(strong["score"], bare["score"] + 0.3)

    def test_weights_sum_to_one_with_skill(self):
        from src.student_profile import HEADLINE_WEIGHTS
        self.assertAlmostEqual(sum(HEADLINE_WEIGHTS.values()), 1.0, places=6)

    def test_college_rigour_credits_research_not_ap(self):
        text = "Undergraduate researcher in the AI lab. Teaching assistant for CS61B. GPA 3.7"
        r = score_application(text, kind=COLLEGE)
        academics = next(d for d in r["dimensions"] if d["key"] == "academics")
        self.assertGreater(academics["score"], 0.5)


class FieldDirectionTests(unittest.TestCase):
    """The fix for "high schooler -> Social Science Research Assistant"."""

    def test_software_skills_point_at_computing(self):
        f = infer_field(["Python", "React", "SQL", "Git", "JavaScript", "Firebase"])
        self.assertEqual(f["name"], "Computing and Mathematics")
        self.assertGreater(f["share"], 0.25)

    def test_field_beats_single_occupation_noise(self):
        f = infer_field(["Python", "Java", "React", "SQL", "Git", "pandas"])
        self.assertNotEqual(f["name"], "Education")

    def test_no_signal_returns_none(self):
        self.assertIsNone(infer_field([]))


class NamingTests(unittest.TestCase):
    def test_vendor_prefixes_and_category_suffixes_are_stripped(self):
        self.assertEqual(pretty_skill("Amazon Web Services AWS SageMaker"), "AWS SageMaker")
        self.assertEqual(pretty_skill("IBM Terraform"), "Terraform")
        self.assertEqual(pretty_skill("Ansible software"), "Ansible")
        self.assertEqual(pretty_skill("Structured query language SQL"), "SQL")

    def test_plain_names_are_untouched(self):
        for name in ("Python", "Snowflake", "Power BI"):
            self.assertEqual(pretty_skill(name), name)


class OccupationMatchTests(unittest.TestCase):
    def test_technical_resume_matches_a_technical_occupation(self):
        ranked = score_soc_candidates(ML_ENGINEER, top_n=5)
        self.assertTrue(ranked)
        self.assertTrue(any(c["soc_code"].startswith("15-") for c in ranked))


if __name__ == "__main__":
    unittest.main()
