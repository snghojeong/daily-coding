"""
Reactive Functional Programming: CLI-Based File Processing

This script demonstrates reactive functional programming to process numbers from a file.
Steps:
1. Reads numbers from a file as a stream of events.
2. Applies functional transformations:
   - Squares each number (`map`).
   - Filters even numbers (`filter`).
   - Sums the results (`reduce`).
3. Outputs the final sum to the CLI.

Features:
- Reactive pipeline with RxPy.
- Pure functions for transformations (`square`, `is_even`, `add`).
- Error handling for missing files or invalid data.
"""

from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
import threading


# Pure function to square a number
# Input: Integer
# Output: Squared value of the input
def square(x: int) -> int:
    return x * x


# Pure function to check if a number is even
# Input: Integer
# Output: Boolean (True if even, False otherwise)
def is_even(x: int) -> bool:
    return x % 2 == 0


# Pure function to sum two numbers
# Inputs: Two integers
# Output: Sum of the inputs
def add(x: int, y: int) -> int:
    return x + y


# Creates an observable stream from a file
# Input: File path as a string
# Output: Observable that emits integers line by line
def file_observable(file_path: str):
    def emitter(observer, _):
        try:
            # Open the file and read each line as an integer
            with open(file_path, 'r') as file:
                for line in file:
                    observer.on_next(int(line.strip()))  # Emit each number
            observer.on_completed()  # Signal the stream is complete
        except FileNotFoundError:
            # Emit an error if the file is missing
            observer.on_error(FileNotFoundError(f"File not found: {file_path}"))
        except ValueError:
            # Emit an error if the file contains invalid data
            observer.on_error(ValueError("Invalid data in file."))
    return create(emitter)


# Sets up a reactive pipeline to process numbers and print results
# Input: File path as a string
# Output: None (results are printed to the CLI)
def reactive_pipeline(file_path: str):
    # Use a thread-based scheduler for asynchronous processing
    thread_scheduler = ThreadPoolScheduler(threading.active_count())

    # Create the reactive pipeline
    file_observable(file_path).pipe(
        ops.map(square),          # Square each number
        ops.filter(is_even),      # Filter even numbers
        ops.reduce(add, seed=0)   # Sum the remaining numbers, starting from 0
    ).subscribe(
        on_next=lambda result: print(f"Final Result: {result}"),  # Print the final sum
        on_error=lambda e: print(f"Error: {e}"),                 # Handle errors
        on_completed=lambda: print("Processing complete."),      # Signal completion
        scheduler=thread_scheduler                              # Use thread-based scheduling
    )


# Configurations
if __name__ == "__main__":
    # Path to the file containing numbers (one per line)
    file_path = "numbers.txt"  # Replace with your file path
    reactive_pipeline(file_path)
