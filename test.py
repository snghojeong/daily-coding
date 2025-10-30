import argparse
from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice


class DataReader(Protocol):
    """Protocol for reading integer data."""
    def read(self) -> Generator[int, None, None]:
        ...


class FileDataReader:
    """Reads integers from a file."""
    def __init__(self, file_path: str, threads: int):
        self.file_path = file_path
        self.threads = threads

    def read(self) -> Generator[int, None, None]:
        try:
            with open(self.file_path, 'rt') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield int(line)
                    except ValueError:
                        print(f"Warning: Invalid number in line: {line}")
        except FileNotFoundError:
            print(f"Error: File not found: {self.file_path}")
        except Exception as e:
            print(f"Error reading file: {e}")


class StringDataReader:
    """Reads integers from a semicolon-separated string."""
    def __init__(self, data_string: str):
        self.data_string = data_string

    def read(self) -> Generator[int, None, None]:
        for part in self.data_string.split(';'):
            part = part.strip()
            if not part:
                continue
            try:
                yield int(part)
            except ValueError:
                print(f"Warning: Invalid number: {part}")


def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n <= 1:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(sqrt(n)) + 1
    return all(n % d != 0 for d in range(3, limit, 2))


def process_chunk(chunk: list[int], transformations: Iterable[Callable[[int], int]] = (lambda x: x * x,)) -> list[int]:
    """Apply transformations and keep primes."""
    processed = []
    for num in chunk:
        val = reduce(lambda acc, f: f(acc), transformations, num)
        if is_prime(val):
            processed.append(val)
    return processed


def sum_primes_threaded(data_reader: DataReader, num_workers: int = 4, chunk_size: int = 4096) -> int:
    """Read, process, and sum primes using threads."""
    total_sum = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        data_iter = data_reader.read()

        while True:
            chunk = list(islice(data_iter, chunk_size))
            if not chunk:
                break
            futures.append(executor.submit(process_chunk, chunk))

        for future in as_completed(futures):
            total_sum += sum(future.result())

    return total_sum


def parse_args():
    parser = argparse.ArgumentParser(description="Sum primes from input data.")
    parser.add_argument("path", help="Path to the input file.")
    parser.add_argument("threads", help="The number of worker threads.")
    return parser.parse_args()


def main():
    args = parse_args()
    reader = FileDataReader(args.path, args.threads)
    result = sum_primes_threaded(reader)
    print(f"Final Result: {result}")


if __name__ == "__main__":
    main()
