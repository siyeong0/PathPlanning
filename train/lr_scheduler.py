from typing import Callable

def linear_schedule(initial_value: float, end_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value + end_value

    return func