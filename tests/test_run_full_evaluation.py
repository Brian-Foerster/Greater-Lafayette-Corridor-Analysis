import unittest
from pathlib import Path

from scripts.run_full_evaluation import _resolve_dynamic_ridership_path


class TestRunFullEvaluationRidershipSource(unittest.TestCase):
    def test_static_mode_ignores_dynamic_path(self) -> None:
        resolved = _resolve_dynamic_ridership_path(
            ridership_source="static",
            dynamic_ridership_path=Path("does/not/matter.csv"),
        )
        self.assertIsNone(resolved)

    def test_auto_mode_uses_existing_dynamic_path(self) -> None:
        path = Path(__file__)
        resolved = _resolve_dynamic_ridership_path(
            ridership_source="auto",
            dynamic_ridership_path=path,
        )
        self.assertEqual(resolved, path)

    def test_auto_mode_falls_back_to_static_when_missing(self) -> None:
        resolved = _resolve_dynamic_ridership_path(
            ridership_source="auto",
            dynamic_ridership_path=Path("missing_dynamic_file.csv"),
        )
        self.assertIsNone(resolved)

    def test_dynamic_mode_requires_existing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _resolve_dynamic_ridership_path(
                ridership_source="dynamic",
                dynamic_ridership_path=Path("missing_dynamic_file.csv"),
            )


if __name__ == "__main__":
    unittest.main()
