import unittest

from scripts.resume_parser import extract_resume_sections


SAMPLE_RESUME = """
JORDAN MITCHELL
Data Analyst

SUMMARY
Data analyst with 3 years of experience in Python and SQL.

EXPERIENCE
Data Analyst — TechRetail Inc.
June 2021 – Present
- Built dashboards in Tableau and PostgreSQL.

EDUCATION
Bachelor of Science in Statistics
University of California, Davis

SKILLS
Python, SQL, Tableau, pandas
"""


class ResumeParserTests(unittest.TestCase):
    def test_extract_resume_sections_returns_expected_categories(self) -> None:
        result = extract_resume_sections(SAMPLE_RESUME)

        self.assertTrue(result["skills"])
        self.assertIn("Python", result["skills"])
        self.assertIn("SQL", result["skills"])
        self.assertIn("Tableau", result["skills"])

        self.assertTrue(result["education"])
        self.assertIn("Bachelor of Science in Statistics", result["education"])
        self.assertIn("University of California, Davis", result["education"])

        self.assertTrue(result["experience"])
        self.assertIn("Data Analyst — TechRetail Inc.", result["experience"])
        self.assertIn("Built dashboards in Tableau and PostgreSQL.", result["experience"])


if __name__ == "__main__":
    unittest.main()
