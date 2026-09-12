import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("apps", "attention_maps", "scripts")
MAX_MODULE_LINES = 1_000


class ModuleSizeBudgetTests(unittest.TestCase):
    def test_application_modules_stay_below_hard_size_limit(self):
        oversized = {}
        for root_name in SOURCE_ROOTS:
            for path in (PROJECT_ROOT / root_name).rglob("*.py"):
                line_count = len(path.read_text(encoding="utf-8").splitlines())
                if line_count > MAX_MODULE_LINES:
                    oversized[str(path.relative_to(PROJECT_ROOT))] = line_count

        self.assertEqual(
            oversized,
            {},
            "Split oversized modules by domain or UI feature before adding more code.",
        )


if __name__ == "__main__":
    unittest.main()
