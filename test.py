from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator, Protocol

class DataReader(Protocol):
    """Protocol for reading data from various sources."""
    def read(self) -> Generator[int, None, None]:
        """Reads data and yields integers."""
        ...

class FileDataReader:
    """Reads data from a file."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self) -> Generator[int, None, None]:
        with open(self.file_path, 'r') as file:
            for line in file:
                try:
                    yield int(line.strip())
                except ValueError:
                    print(f"Warning: Invalid number in line: {line}")

class StringDataReader:
    """Reads data from a string."""
    def __init__(self, data_string: str):
        self.data_string = data_string

    def read(self) -> Generator[int, None, None]:
        for number in self.data_string.split(','):
            try:
                yield int(number)
            except ValueError:
                print(f"Warning: Invalid number: {number}")

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

def sum_primes(data_reader: DataReader) -> int:
    """
    Reads data, applies transformations, filters primes, and sums the results.

    Args:
        data_reader: An object that implements the DataReader protocol.

    Returns:
        The sum of prime numbers after transformations.
    """
    return sum(process_data(data_reader.read()))

def main():
    """Main function to process data and print the results."""
    # Example usage:
    file_data_reader = FileDataReader("file:///path/to/numbers.txt") 
    string_data_reader = StringDataReader("1,2,3,4,5") 

    result_file = sum_primes(file_data_reader)
    result_string = sum_primes(string_data_reader)

    print(f"Result from file: {result_file}")
    print(f"Result from string: {result_string}")

if __name__ == "__main__":
    main()
