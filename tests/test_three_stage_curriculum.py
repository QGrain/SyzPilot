import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.curriculum import (
    DENSE_PAPER_MODE,
    SPARSE_ADAPTIVE_MODE,
    curriculum_class,
    curriculum_metric_class_members,
    curriculum_output_classes,
    infer_curriculum_stage,
    stage3_active_classes,
)


class ThreeStageCurriculumTest(unittest.TestCase):
    def test_progression_uses_ternary_stage_before_exact_stage(self):
        self.assertEqual(
            infer_curriculum_stage(
                {0: 900, 1: 50, 2: 50, 3: 0, 4: 0},
                5,
                minimum_total=1000,
                minimum_positive_class=100,
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            1,
        )
        self.assertEqual(
            infer_curriculum_stage(
                {0: 700, 1: 100, 2: 100, 3: 100, 4: 0},
                5,
                minimum_total=1000,
                minimum_positive_class=100,
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            2,
        )
        self.assertEqual(
            infer_curriculum_stage(
                {0: 600, 1: 150, 2: 150, 3: 0, 4: 100},
                5,
                minimum_total=1000,
                minimum_positive_class=100,
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            3,
        )
        self.assertEqual(
            infer_curriculum_stage(
                {0: 900, 1: 0, 2: 0, 3: 0, 4: 100},
                5,
                minimum_total=1000,
                minimum_positive_class=100,
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            1,
        )

    def test_sparse_intermediate_class_does_not_block_stage_three(self):
        active = stage3_active_classes(
            {0: 600, 1: 150, 2: 150, 3: 0, 4: 100},
            5,
            minimum_class_samples=100,
            curriculum_mode=SPARSE_ADAPTIVE_MODE,
        )
        self.assertEqual(active, (0, 1, 2, 4))
        self.assertEqual(curriculum_class(4, 5, 3, active), 3)
        self.assertEqual(curriculum_class(3, 5, 3, active), 4)
        self.assertEqual(curriculum_output_classes(5, 3, active), 5)
        self.assertEqual(
            curriculum_metric_class_members(5, 3, active)[4], (3,)
        )

    def test_active_classes_are_monotonic_and_require_split_evidence(self):
        self.assertEqual(
            stage3_active_classes(
                {0: 700, 1: 120, 2: 100, 3: 10, 4: 100},
                5,
                minimum_class_samples=100,
                train_counts={0: 600, 1: 100, 2: 80, 3: 10, 4: 80},
                validation_counts={0: 100, 1: 20, 2: 20, 3: 0, 4: 20},
                previous_active_classes=(0, 1, 4),
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            (0, 1, 2, 4),
        )
        self.assertEqual(
            stage3_active_classes(
                {0: 700, 1: 120, 2: 100, 3: 10, 4: 100},
                5,
                minimum_class_samples=100,
                train_counts={0: 600, 1: 100, 2: 80, 3: 10, 4: 100},
                validation_counts={0: 100, 1: 20, 2: 20, 3: 0, 4: 0},
                previous_active_classes=(0, 1, 4),
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            (),
        )
        self.assertEqual(
            stage3_active_classes(
                {0: 700, 1: 120, 2: 100, 3: 10, 4: 100},
                5,
                minimum_class_samples=100,
                train_counts={0: 600, 1: 100, 2: 100, 3: 10, 4: 80},
                validation_counts={0: 100, 1: 20, 2: 0, 3: 0, 4: 20},
                previous_active_classes=(0, 1, 4),
                curriculum_mode=SPARSE_ADAPTIVE_MODE,
            ),
            (0, 1, 4),
        )

    def test_dense_mode_requires_every_exact_class(self):
        sparse_counts = {0: 600, 1: 150, 2: 150, 3: 0, 4: 100}
        self.assertEqual(
            infer_curriculum_stage(
                sparse_counts,
                5,
                minimum_total=1000,
                minimum_positive_class=100,
                curriculum_mode=DENSE_PAPER_MODE,
            ),
            2,
        )
        self.assertEqual(
            stage3_active_classes(
                sparse_counts,
                5,
                minimum_class_samples=100,
                curriculum_mode=DENSE_PAPER_MODE,
            ),
            (),
        )
        dense_counts = {0: 600, 1: 100, 2: 100, 3: 100, 4: 100}
        self.assertEqual(
            infer_curriculum_stage(
                dense_counts,
                5,
                minimum_total=1000,
                minimum_positive_class=100,
                curriculum_mode=DENSE_PAPER_MODE,
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
