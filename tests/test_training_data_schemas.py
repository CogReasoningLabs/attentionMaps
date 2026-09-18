import unittest

from attention_maps.datasets import (
    STANDARD_TRAINING_SCHEMAS,
    TASK_SPECIFIC_SUPERVISED_SCHEMA,
    infer_training_schema,
)


class TrainingDataSchemaTests(unittest.TestCase):
    def test_defines_exactly_four_standard_training_schemas(self):
        self.assertEqual(len(STANDARD_TRAINING_SCHEMAS), 4)
        self.assertEqual(
            {schema.key for schema in STANDARD_TRAINING_SCHEMAS},
            {
                "pretraining",
                "instruction_finetuning",
                "task_specific_supervised",
                "preference_tuning",
            },
        )

    def test_maps_task_specific_catalog_data_to_d2_eligible_schema(self):
        self.assertEqual(
            infer_training_schema("Task-specific fine-tuning"),
            TASK_SPECIFIC_SUPERVISED_SCHEMA,
        )
        self.assertIsNone(infer_training_schema("Unclassified — pending review"))


if __name__ == "__main__":
    unittest.main()
