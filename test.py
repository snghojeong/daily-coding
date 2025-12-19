import argparse
from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator, Protocol, Optional # Added Optional for clarity in DataReader return type
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice


class DataReader(Protocol):
    def read(self) -> Generator[int, None, None]:
        ...


class FileDataReader:
    def __init__(self, file_path: str):
        self.file_path = file_path

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
                        print(f"Warning: Invalid number in line: **{line}**") # Bolding for emphasis
        except FileNotFoundError:
            # Raising an error is often better than just printing, but keeping it simple
            print(f"Error: File not found: **{self.file_path}**")
        except Exception as e:
            print(f"Error reading file: **{e}**")


class StringDataReader:
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
                print(f"Warning: Invalid number: **{part}**")


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(sqrt(n)) + 1
    return all(n % d != 0 for d in range(3, limit, 2))


def process_chunk(chunk: list[int], transformations: Iterable[Callable[[int], int]] = (lambda x: x * x,)) -> list[int]:
    processed = []
    # Using a list comprehension for the transformation part is often more Pythonic
    for num in chunk:
        val = reduce(lambda acc, f: f(acc), transformations, num)
        if is_prime(val):
            processed.append(val)
    return processed


def sum_primes_threaded(data_reader: DataReader, num_workers: int, chunk_size: int) -> int:
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
            # Added a try-except block to handle potential exceptions in worker threads,
            # though not strictly required by the prompt, it's a good practice.
            try:
                total_sum += sum(future.result())
            except Exception as exc:
                print(f'Chunk processing generated an exception: {exc}')

    return total_sum


def parse_args():
    parser = argparse.ArgumentParser(description="Sum primes from input data.")
    parser.add_argument("path", help="Path to the input file.")
    # --- START OF KEY MODIFICATIONS ---
    # 1. Added 'type=int' for numerical arguments. They are read as strings by default.
    parser.add_argument("threads", type=int, help="The number of worker threads.")
    parser.add_argument("chunk_size", type=int, help="The size of a chunk.")
    # 2. Changed 'debug' to a store_true action for a proper boolean flag.
    #    This means you use --debug (or simply 'debug' if you want it positional) without an argument.
    parser.add_argument("--debug", action="store_true", help="Print debug message.")
    # --- END OF KEY MODIFICATIONS ---
    return parser.parse_args()


def main():
    args = parse_args()
    # Check for threads/chunk_size validity (e.g., must be positive) is recommended,
    # but not implemented for simplicity.
    reader = FileDataReader(args.path)
    result = sum_primes_threaded(reader, args.threads, args.chunk_size)
    print(f"Final Result: **{result}**")


if __name__ == "__main__":
    main()
