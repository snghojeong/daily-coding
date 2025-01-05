from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
from rx.disposable import Disposable
import threading
import math
from typing import Callable, Iterable, Iterator

# Create a thread pool scheduler
thread_pool_scheduler = ThreadPoolScheduler(threading.active_count())

def is_prime(x: int) -> bool:
    """Checks if a number is prime."""
    if x < 2:
        return False
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True

def read_numbers_from_file(file_path: str) -> Iterator[int]:
    """Reads numbers from a file and yields them."""
    with open(file_path, 'r') as file:
        for line in file:
            try:
                yield int(line.strip())
            except ValueError:
                # Log the invalid line instead of raising an error
                print(f"Warning: Invalid number in line: {line}")

def process_numbers(
    source: Iterable[int],
    transformations: Iterable[Callable[[int], int]] = (lambda x: x * x,),  # Square by default
    filter_predicate: Callable[[int], bool] = is_prime,
    aggregation_function: Callable[[int, int], int] = sum,  # Use built-in sum
) -> int:
    """
    Processes a stream of numbers with the specified transformations, filtering, and aggregation.

    Args:
        source: An iterable of numbers.
        transformations: An iterable of functions to apply to each number.
        filter_predicate: A function to filter numbers.
        aggregation_function: A function to aggregate the filtered numbers.

    Returns:
        The aggregated result.
    """
    return sum(
        num
        for num in map(
            lambda x: reduce(lambda acc, f: f(acc), transformations, x), source
        )
        if filter_predicate(num)
    )

if __name__ == "__main__":
    file_path = "numbers.txt"
    result = process_numbers(read_numbers_from_file(file_path))
    print(f"Final Result: {result}") 
