import unittest
from resources.resource_catalog import get_resources, CURATED_RESOURCES


class TestResourceCatalog(unittest.TestCase):
    def test_all_six_skills_present(self):
        skills = ["Python", "SQL", "Power BI", "Machine Learning", "scikit-learn", "Docker"]
        for s in skills:
            with self.subTest(skill=s):
                res = get_resources(s)
                self.assertGreater(len(res), 0)

    def test_alias_resolution(self):
        self.assertEqual(get_resources("powerbi"), get_resources("Power BI"))
        self.assertEqual(get_resources("sklearn"), get_resources("scikit-learn"))
        self.assertEqual(get_resources("ml"), get_resources("Machine Learning"))
        self.assertEqual(get_resources("python"), get_resources("Python"))

    def test_unknown_skill(self):
        self.assertEqual(get_resources("UnknownSkill123"), [])

    def test_empty_and_whitespace(self):
        self.assertEqual(get_resources(""), [])
        self.assertEqual(get_resources("   "), [])

    def test_non_string_type_error(self):
        for invalid in [None, 123, 4.5, ["Python"]]:
            with self.subTest(val=invalid):
                with self.assertRaises(TypeError):
                    get_resources(invalid)

    def test_resource_entry_structure(self):
        required_keys = {"title", "url", "description", "source_type"}
        for skill, items in CURATED_RESOURCES.items():
            for item in items:
                self.assertEqual(set(item.keys()), required_keys)
                for k, v in item.items():
                    self.assertTrue(isinstance(v, str) and len(v.strip()) > 0)

    def test_catalog_immutability(self):
        res = get_resources("Python")
        self.assertGreater(len(res), 0)
        original_title = res[0]["title"]
        res[0]["title"] = "Hacked Title"
        fresh_res = get_resources("Python")
        self.assertEqual(fresh_res[0]["title"], original_title)


if __name__ == "__main__":
    unittest.main()
