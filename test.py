from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
import threading
import math

# Create a thread pool scheduler
thread_pool_scheduler = ThreadPoolScheduler(threading.active_count())

# Pure function to check if a number is prime
def is_prime(x: int) -> bool:
    if x < 2:
        return False
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True

# Pure function to square a number
def square(x: int) -> int:
    return x * x

# Pure function to sum two numbers
def add(x: int, y: int) -> int:
    return x + y

# Creates an observable stream from a file
def create_number_observable(file_path: str) -> rx.Observable:
    def emitter(observer, _):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    observer.on_next(int(line.strip()))
            observer.on_completed()
        except (IOError, OSError) as e:
            observer.on_error(e)
        except ValueError:
            observer.on_error(ValueError("Invalid data in file."))
    return create(emitter)

# Defines the processing pipeline for the number stream
def process_numbers(source: rx.Observable) -> rx.Disposable:
    return source.pipe(
        ops.map(square),
        ops.filter(is_prime),
        ops.reduce(add, seed=0)
    ).subscribe(
        on_next=lambda result: print(f"Final Result: {result}"),
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print("Processing complete."),
        scheduler=thread_pool_scheduler
    )

if __name__ == "__main__":
    file_path = "numbers.txt"  # Replace with your file path
    number_observable = create_number_observable(file_path)
    process_numbers(number_observable)
