import csv
import json
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase


class RetrieverEvaluator:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path

        self.metrics = {
            "contextual_precision": ContextualPrecisionMetric(),
            "contextual_recall": ContextualRecallMetric(),
            "contextual_relevancy": ContextualRelevancyMetric(),
        }

        # Write CSV header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "input",
                "actual_output",
                "expected_output",
                "retrieval_context",
                "metric",
                "score",
                "reason",
            ])

    def evaluate(self, data: dict):
        """
        data = {
            "input": str,
            "actual_output": str,
            "expected_output": str,
            "retrieval_context": List[str]
        }
        """

        test_case = LLMTestCase(
            input=data["input"],
            actual_output=data["actual_output"],
            expected_output=data["expected_output"],
            retrieval_context=data["retrieval_context"],
        )

        serialized_context = json.dumps(
            data["retrieval_context"], ensure_ascii=False
        )

        rows = []

        for name, metric in self.metrics.items():
            metric.measure(test_case)
            rows.append([
                data["input"],
                data["actual_output"],
                data["expected_output"],
                serialized_context,
                name,
                metric.score,
                metric.reason,
            ])

        # Append results
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
