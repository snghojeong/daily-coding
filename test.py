import argparse
from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from itertools import islice # import islice

class DataReader(Protocol):
    """Protocol for reading data from various sources."""
    def read(self) -> Generator[int, None, None]:
        """Abstract function to read data and yields integers."""
        ...

class FileDataReader:
    """Reads data from a file."""
    def __init__(self, file_path: str):
        """Initializes FileDataReader with the file path."""
        self.file_path = file_path

    def read(self) -> Generator[int, None, None]:
        """Reads integers from the file."""
        try:
            with open(self.file_path, 'rt') as file:
                for line in file:
                    try:
                        yield int(line.strip())
                    except ValueError:
                        print(f"Warning: Invalid number in line: {line.strip()}")  # Show the line content
        except FileNotFoundError:
            print(f"Error: File not found: {self.file_path}")
        except Exception as e:  # Catch other potential file errors
            print(f"Error reading file: {e}")

class StringDataReader:
    """Reads data from a string."""
    def __init__(self, data_string: str):
        """Initializes StringDataReader with the data string."""
        self.data_string = data_string

    def read(self) -> Generator[int, None, None]:
        """Reads integers from the string."""
        for number in self.data_string.split(';'):
            try:
                yield int(number.strip()) #strip in case of whitespaces in the string
            except ValueError:
                print(f"Warning: Invalid number: {number.strip()}") #strip in case of whitespaces in the string

def is_prime(number: int) -> bool:
    """Checks if a number is prime (optimized)."""
    if number <= 1:
        return False
    if number % 2 == 0 and number > 2:
        return False
    for divisor in range(3, int(sqrt(number)) + 1, 2):
        if number % divisor == 0:
            return False
    return True

def process_chunk(chunk: list[int], transformations: Iterable[Callable] = (lambda x: x * x,)) -> list[int]:
    """Processes a chunk of data with transformations and filtering."""
    processed_chunk = []
    for num in chunk:
        transformed = reduce(lambda acc, f: f(acc), transformations, num)
        if is_prime(transformed):
            processed_chunk.append(transformed)
    return processed_chunk

def sum_primes_threaded(data_reader: DataReader, num_workers: int = 4) -> int:
    """Reads data, processes it in parallel, and sums prime numbers."""

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        chunk_size = 1024  # Adjust chunk size as needed

        data_iter = data_reader.read()

        while True:
            chunk = list(islice(data_iter, chunk_size)) # use islice so that we can read from generator in chunks
            if not chunk:
                break
            futures.append(executor.submit(process_chunk, chunk))

        total_sum = 0
        for future in as_completed(futures):
            total_sum += sum(future.result())
        return total_sum
        
def main():
    """Main function to process data and print the results."""
    args = parse_args()
    """ no need for file://, its assumed its a local file """
    file_data_reader = FileDataReader(args.path) 
    result = sum_primes_threaded(file_data_reader)
    print(f"Final Result: {result}")

if __name__ == "__main__":
    main()

