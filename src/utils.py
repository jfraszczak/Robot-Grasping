import time
from typing import Callable, Any


def measure_time(func: Callable) -> None:
    def wrapper(*args, **kwargs) -> Any:
        start: float = time.perf_counter()
        result: Any = func(*args, **kwargs)
        end: float = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f}s")
        return result
    return wrapper
