from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator
import argparse

def is_prime(number: int) -> bool:
    """Checks if a number is prime."""
    if number <= 1:
        return False
    if number % 2 == 0 and number > 2:
        return False
    for divisor in range(3, int(sqrt(number)) + 1, 2):
        if number % divisor == 0:
            return False
    return True

def read_numbers_from_file(file_path: str) -> Generator[int, None, None]:
    """Reads numbers from a file and yields them."""
    with open(file_path, 'r') as file:
        for line in file:
            try:
                yield int(line.strip())
            except ValueError:
                print(f"Warning: Invalid number in line: {line}")

def process_data(
    data: Iterable,
    transformations: Iterable[Callable] = (lambda x: x * x,),  # Square by default
    filter_predicate: Callable[[any], bool] = is_prime,
) -> Iterable:
    """
    Processes an iterable of data with the specified transformations and filtering.

    Args:
        data: An iterable of data.
        transformations: An iterable of functions to apply to each data element.
        filter_predicate: A function to filter data elements.

    Returns:
        An iterable of processed data.
    """
    return filter(
        filter_predicate,
        map(
            lambda x: reduce(lambda acc, f: f(acc), transformations, x),
            data,
        ),
    )

def sum_primes(file_path: str) -> int:
    """
    Reads numbers from a file, applies transformations, filters primes, and sums the results.

    Args:
        file_path: Path to the file containing numbers.

    Returns:
        The sum of prime numbers after transformations.
    """
    return sum(process_data(read_numbers_from_file(file_path)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sum prime numbers from a file.")
    parser.add_argument("file_path", help="Path to the input file")
    args = parser.parse_args()

    result = sum_primes(args.file_path)
    print(f"Final Result: {result}")
