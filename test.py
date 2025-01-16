from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def read_data(source: str) -> Generator[int, None, None]:
    """Reads data from the specified source.

    Args:
        source: The source of the data. Can be a file path or a string containing numbers.

    Returns:
        A generator yielding the extracted integers.
    """
    if source.startswith("file://"):
        file_path = source[7:]  # Remove "file://" prefix
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    yield int(line.strip())
                except ValueError:
                    print(f"Warning: Invalid number in line: {line}")
    else:
        for number in source.split(','):
            try:
                yield int(number)
            except ValueError:
                print(f"Warning: Invalid number: {number}")

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

def sum_primes_with_threads(data: Iterable, num_workers: int = 4) -> int:
    """
    Processes data in parallel using a thread pool and sums the prime numbers.

    Args:
        data: An iterable of data.
        num_workers: The number of worker threads in the thread pool.

    Returns:
        The sum of prime numbers after processing.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(sum, process_data(chunk)) for chunk in chunks(data, num_workers)]
        return sum(f.result() for f in as_completed(futures))

def chunks(data, n):
    """Yield successive n-sized chunks from data."""
    for i in range(0, len(data), n):
        yield data[i:i + n]

def main():
    """Main function to process data and print the results."""
    data_source = "file:///path/to/numbers.txt"  # Or "1,2,3,4,5" 
    result = sum_primes_with_threads(read_data(data_source))
    print(f"Final Result: {result}")

if __name__ == "__main__":
    main()
