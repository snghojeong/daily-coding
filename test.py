from functools import reduce
from math import sqrt
from typing import Callable, Iterable

def is_prime(number: int) -> bool:
    """Checks if a number is prime."""
    if number < 2:
        return False
    for divisor in range(2, int(sqrt(number)) + 1):
        if number % divisor == 0:
            return False
    return True

def read_numbers_from_file(file_path: str) -> Iterable[int]:
    """Reads numbers from a file and yields them."""
    with open(file_path, 'r') as file:
        for line in file:
            try:
                yield int(line.strip())
            except ValueError:
                print(f"Warning: Invalid number in line: {line}")

def process_numbers(
    numbers: Iterable[int],
    transformations: Iterable[Callable[[int], int]] = (lambda x: x * x,),  # Square by default
    filter_predicate: Callable[[int], bool] = is_prime,
) -> int:
    """
    Processes a stream of numbers with the specified transformations and filtering.

    Args:
        numbers: An iterable of numbers.
        transformations: An iterable of functions to apply to each number.
        filter_predicate: A function to filter numbers.

    Returns:
        The sum of the processed numbers.
    """
    return sum(
        num
        for num in map(
            lambda x: reduce(lambda acc, f: f(acc), transformations, x), numbers
        )
        if filter_predicate(num)
    )

if __name__ == "__main__":
    file_path = "numbers.txt"
    result = process_numbers(read_numbers_from_file(file_path))
    print(f"Final Result: {result}")
