"""
Reactive Functional Programming: CLI-Based File Processing

This script uses RxPy for reactive file processing. It:
1. Reads numbers from a file as a stream of events.
2. Applies transformations:
   - Squares numbers (`map`).
   - Filters even numbers (`filter`).
   - Sums the results (`reduce`).
3. Outputs the final sum to the CLI.

Features:
- Pure functions (`square`, `is_even`, `add`).
- Reactive pipeline with RxPy operators.
- Error handling for missing files or invalid data.
"""

from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
import threading


def square(x: int) -> int:
    return x * x


def is_even(x: int) -> bool:
    return x % 2 == 0


def add(x: int, y: int) -> int:
    return x + y


# Creates an observable stream from a file
def file_observable(file_path: str):
    def emitter(observer, _):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    observer.on_next(int(line.strip()))
            observer.on_completed()
        except FileNotFoundError:
            observer.on_error(FileNotFoundError(f"File not found: {file_path}"))
        except ValueError:
            observer.on_error(ValueError("Invalid data in file."))
    return create(emitter)


# Processes numbers reactively and prints results to the CLI
def reactive_pipeline(file_path: str):
    thread_scheduler = ThreadPoolScheduler(threading.active_count())
    file_observable(file_path).pipe(
        ops.map(square),
        ops.filter(is_even),
        ops.reduce(add, seed=0)
    ).subscribe(
        on_next=lambda result: print(f"Final Result: {result}"),
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print("Processing complete."),
        scheduler=thread_scheduler
    )


# Configurations
if __name__ == "__main__":
    file_path = "numbers.txt"
    reactive_pipeline(file_path)
