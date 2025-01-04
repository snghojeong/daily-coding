from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
from rx.disposable import Disposable
import threading
import math
from typing import Callable, Iterable

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

def create_number_observable(file_path: str) -> rx.Observable:
    """Creates an observable stream from a file."""
    def emitter(observer, _):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    try:
                        observer.on_next(int(line.strip()))
                    except ValueError:
                        observer.on_error(ValueError(f"Invalid number in line: {line}"))
            observer.on_completed()
        except (IOError, OSError) as e:
            observer.on_error(e)
    return create(emitter)

def process_numbers(
    source: rx.Observable,
    transformations: Iterable[Callable[[int], int]] = (square,),
    filter_predicate: Callable[[int], bool] = is_prime,
    aggregation_function: Callable[[int, int], int] = add,
    initial_value: int = 0,
) -> rx.Disposable:
    """
    Processes a stream of numbers with the specified transformations, filtering, and aggregation.

    Args:
        source: The source observable stream of numbers.
        transformations: An iterable of functions to apply to each number.
        filter_predicate: A function to filter numbers.
        aggregation_function: A function to aggregate the filtered numbers.
        initial_value: The initial value for the aggregation.

    Returns:
        A disposable object representing the subscription.
    """
    return source.pipe(
        ops.map(lambda x: reduce(lambda acc, f: f(acc), transformations, x)),
        ops.filter(filter_predicate),
        ops.reduce(aggregation_function, initial_value)
    ).subscribe(
        on_next=lambda result: print(f"Final Result: {result}"),
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print("Processing complete."),
        scheduler=thread_pool_scheduler
    )

if __name__ == "__main__":
    file_path = "numbers.txt"
    number_observable = create_number_observable(file_path)
    process_numbers(number_observable)
