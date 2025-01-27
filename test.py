from functools import reduce
from math import sqrt
from typing import Callable, Iterable, Generator, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

class DataReader(Protocol):
    """Protocol for reading data from various sources."""
    def read(self) -> Generator[int, None, None]:
        ...

class FileDataReader:
    """Reads data from a file."""
    def __init__(self, file_path: str):
        """
        Initializes the FileDataReader with the path to the file.

        Args:
            file_path: Path to the file containing the data.
        """
        self.file_path = file_path

    def read(self) -> Generator[int, None, None]:
        """
        Reads numbers from the file and yields them.

        Returns:
            A generator that yields integers read from the file.
        """
        with open(self.file_path, 'r') as file:
            for line in file:
                try:
                    yield int(line.strip())
                except ValueError:
                    print(f"Warning: Invalid number in line: {line}")

class StringDataReader:
    """Reads data from a string."""
    def __init__(self, data_string: str):
        """
        Initializes the StringDataReader with the data string.

        Args:
            data_string: String containing numbers separated by commas.
        """
        self.data_string = data_string

    def read(self) -> Generator[int, None, None]:
        """
        Reads numbers from the string and yields them.

        Returns:
            A generator that yields integers extracted from the string.
        """
        for number in self.data_string.split(','):
            try:
                yield int(number)
            except ValueError:
                print(f"Warning: Invalid number: {number}")

def is_prime(number: int) -> bool:
    """
    Checks if a number is prime.

    Args:
        number: The number to check for primality.

    Returns:
        True if the number is prime, False otherwise.
    """
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
    return filter(filter_predicate, 
                  map(lambda x: reduce(lambda acc, f: f(acc), transformations, x), data)) 

def sum_primes_with_thread_pool(data_reader: DataReader, num_workers: int = 4) -> int:
    """
    Reads data, processes it in parallel, filters primes, and sums the results.

    Args:
        data_reader: An object that implements the DataReader protocol.
        num_workers: The number of worker threads in the thread pool.

    Returns:
        The sum of prime numbers after transformations.
    """
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        data_queue = Queue()
        results_queue = Queue()

        def worker():
            """Worker function for processing data chunks."""
            while True:
                try:
                    chunk = data_queue.get_nowait()
                except Empty:
                    break
                results_queue.put(sum(process_data(chunk)))

        for _ in range(num_workers):
            executor.submit(worker)

        for chunk in chunks(data_reader.read(), num_workers):
            data_queue.put(chunk)

        data_queue.join() 

        result = 0
        while not results_queue.empty():
            result += results_queue.get()

        return result

def chunks(data, n):
    """
    Yield successive n-sized chunks from data.

    Args:
        data: The iterable to be chunked.
        n: The size of each chunk.

    Yields:
        Successive n-sized chunks from the data.
    """
    for i in range(0, len(data), n):
        yield data[i:i + n]

def main():
    """
    Main function to process data and print the results.
    """
    file_data_reader = FileDataReader("file:///path/to/numbers.txt") 
    result = sum_primes_with_thread_pool(file_data_reader)
    print(f"Final Result: {result}")

if __name__ == "__main__":
    main()
