"""
Reactive Functional Programming: CLI-Based File Processing (Prime Numbers)

This script demonstrates reactive functional programming to process numbers from a file.
Steps:
1. Reads numbers from a file as a stream of events.
2. Applies functional transformations:
   - Squares each number (`map`).
   - Filters prime numbers (`filter`).
   - Sums the results (`reduce`).
3. Outputs the final sum to the CLI.

"""

from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
import threading
import math


# Pure function to square a number
def square(x: int) -> int:
    return x * x


# Pure function to check if a number is prime
def is_prime(x: int) -> bool:
    if x < 2:
        return False
    for i in range(2, int(math.sqrt(x)) + 1):
        if x % i == 0:
            return False
    return True


# Pure function to sum two numbers
def add(x: int, y: int) -> int:
    return x + y

def sub(x: int, y: int) -> int:
    return x - y

# Creates an observable stream from a file
def file_observable(file_path: str):
    def emitter(observer, _):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    observer.on_next(int(line.strip()))  # Emit each number
            observer.on_completed()  # Signal the stream is complete
        except FileNotFoundError:
            observer.on_error(FileNotFoundError(f"File not found: {file_path}"))
        except ValueError:
            observer.on_error(ValueError("Invalid data in file."))
    return create(emitter)


# Sets up a reactive pipeline to process numbers and print results
def reactive_pipeline(file_path: str):
    thread_scheduler = ThreadPoolScheduler(threading.active_count())

    file_observable(file_path).pipe(
        ops.map(square),          # Square each number
        ops.filter(is_prime),     # Filter prime numbers
        ops.reduce(add, seed=0)   # Sum the remaining numbers, starting from 0
    ).subscribe(
        on_next=lambda result: print(f"Final Result: {result}"),  # Print the final sum
        on_error=lambda e: print(f"Error: {e}"),                 # Handle errors
        on_completed=lambda: print("Processing complete."),      # Signal completion
        scheduler=thread_scheduler                              # Use thread-based scheduling
    )

if __name__ == "__main__":
    file_path = "numbers.txt"  # Replace with your file path
    reactive_pipeline(file_path)
