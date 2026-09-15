from evaluation.aggregator import EvaluationMetricsAggregator
from evaluation.models.cases import EvaluationCase
from evaluation.models.reports import (
    CategoryEvaluation,
    CategoryEvaluationReport,
    EvaluationCaseResult,
)


class CategoryEvaluationBuilder:
    def __init__(self):
        self.aggregator = EvaluationMetricsAggregator()

    def build(
        self,
        cases: list[EvaluationCase],
        results: list[EvaluationCaseResult],
    ) -> CategoryEvaluationReport:

        if not cases:
            raise ValueError("cases cannot be empty")

        if not results:
            raise ValueError("results cannot be empty")

        case_id_to_case = {case.id: case for case in cases}

        for result in results:
            if result.case_id not in case_id_to_case:
                raise ValueError(f"{result.case_id} in not found in given cases.")

        group_category = {}

        for result in results:
            category = case_id_to_case[result.case_id].category

            group_category.setdefault(category, []).append(result)

        categories = []

        for category, category_results in group_category.items():
            agg_result = self.aggregator.aggregate(category_results)

            category_eval = CategoryEvaluation(
                total_cases=len(category_results),
                category=category,
                **agg_result,
            )

            categories.append(category_eval)

        return CategoryEvaluationReport(total_cases=len(cases), categories=categories)
