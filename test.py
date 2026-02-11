import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from math import isqrt

def get_numbers_from_file(file_path: str):
    try:
        with open(file_path, 'rt') as file:
            for line in file:
                if stripped := line.strip():
                    try:
                        yield int(stripped)
                    except ValueError:
                        print(f"Warning: Invalid number: {stripped}")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")

def is_prime(n: int) -> bool:
    """Efficiently checks if a number is prime."""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    # Check divisors up to sqrt(n)
    for i in range(5, isqrt(n) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

def process_chunk(chunk: list[int]) -> int:
    """Transforms numbers and returns the sum of those that are prime."""
    # Applying the transformation (x*x) and filtering for primes in one pass
    transformed = (x * x for x in chunk)
    return sum(val for val in transformed if is_prime(val))

def sum_primes_parallel(numbers_iter, num_workers: int, chunk_size: int) -> int:
    """Distributes chunks to a thread pool and aggregates results."""
    total_sum = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create a generator of futures to save memory
        def chunk_generator():
            while True:
                chunk = list(islice(numbers_iter, chunk_size))
                if not chunk:
                    break
                yield executor.submit(process_chunk, chunk)

        for future in chunk_generator():
            try:
                total_sum += future.result()
            except Exception as e:
                print(f"Chunk failed: {e}")
    
    return total_sum

def main():
    parser = argparse.ArgumentParser(description="Sum squares of primes from a file.")
    parser.add_argument("path", help="Path to input file")
    parser.add_argument("threads", type=int, help="Number of threads")
    parser.add_argument("chunk_size", type=int, help="Chunk size")
    args = parser.parse_args()

    numbers = get_numbers_from_file(args.path)
    result = sum_primes_parallel(numbers, args.threads, args.chunk_size)
    print(f"Final Result: {result}")

if __name__ == "__main__":
    main()
