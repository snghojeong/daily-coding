from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator
import sys

def is_prime(number: int) -> bool:
    if number <= 1:
        return False
    if number % 2 == 0 and number > 2:
        return False
    for divisor in range(3, int(sqrt(number)) + 1, 2): 
        if number % divisor == 0:
            return False
    return True

def read_numbers_from_file(file_path: str) -> Generator[int, None, None]:
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
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    result = sum_primes(file_path)
    print(f"Final Result: {result}")
