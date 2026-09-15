import statistics


def calculate_p95(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty")

    if len(values) < 2:
        return values[0]

    return statistics.quantiles(values, n=100)[94]
