from rx import create, operators as ops
from rx.scheduler import ThreadPoolScheduler
import threading


# Step 1: Define Pure Functions for Transformations
def square(x: int) -> int:
    """Pure function to square a number."""
    return x * x


def is_even(x: int) -> bool:
    """Pure function to check if a number is even."""
    return x % 2 == 0


def add(x: int, y: int) -> int:
    """Pure function to sum two numbers."""
    return x + y


# Step 2: Reactive Source - Read Numbers from File
def file_observable(file_path: str):
    """Creates an observable stream from a file."""
    def emitter(observer, _):
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    observer.on_next(int(line.strip()))  # Emit each number
            observer.on_completed()
        except FileNotFoundError:
            observer.on_error(FileNotFoundError(f"File not found: {file_path}"))
        except ValueError:
            observer.on_error(ValueError("File contains invalid data. Ensure all lines are integers."))

    return create(emitter)


# Step 3: Reactive Pipeline for CLI Output
def reactive_pipeline(file_path: str):
    """Processes numbers reactively and prints the result to the CLI."""
    # Use a thread pool scheduler for reactive asynchronous processing
    thread_scheduler = ThreadPoolScheduler(threading.active_count())

    # Reactive file source
    file_observable(file_path).pipe(
        ops.map(square),                 # Map: Square each number
        ops.filter(is_even),             # Filter: Keep only even numbers
        ops.reduce(add, seed=0)          # Reduce: Compute the sum with an initial value of 0
    ).subscribe(
        on_next=lambda result: print(f"Final Result: {result}"),
        on_error=lambda e: print(f"Error: {e}"),
        on_completed=lambda: print("Processing complete."),
        scheduler=thread_scheduler
    )


# Configurations
if __name__ == "__main__":
    file_path = "numbers.txt"  # Replace with your file path
    reactive_pipeline(file_path)
