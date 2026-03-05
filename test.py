import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from math import isqrt
from typing import Iterable, Iterator, List

log = logging.getLogger(__name__)


def read_ints(path: str) -> Iterator[int]:
    try:
        with open(path, "rt", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield int(s)
                except ValueError:
                    log.warning("Invalid number: %r", s)
    except FileNotFoundError:
        log.error("File not found: %s", path)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3

    limit = isqrt(n)
    d = 5
    while d <= limit:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def chunked(it: Iterable[int], size: int) -> Iterator[List[int]]:
    """Yield lists of up to `size` items from an iterable."""
    if size <= 0:
        raise ValueError("chunk size must be > 0")

    it = iter(it)
    return iter(lambda: list(islice(it, size)), [])


def sum_prime_squares(nums: List[int]) -> int:
    """Sum x^2 for x in nums where x^2 is prime (same logic as original)."""
    total = 0
    for x in nums:
        sq = x * x
        if is_prime(sq):
            total += sq
    return total


def sum_primes_parallel(numbers: Iterable[int], workers: int, chunk_size: int) -> int:
    if workers <= 0:
        raise ValueError("workers must be > 0")

    total = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(sum_prime_squares, c) for c in chunked(numbers, chunk_size)]
        for fut in as_completed(futures):
            try:
                total += fut.result()
            except Exception:
                log.exception("Chunk failed")
    return total


@dataclass(frozen=True)
class Args:
    path: str
    threads: int
    chunk_size: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Sum squares of primes from a file.")
    p.add_argument("path", help="Path to input file")
    p.add_argument("threads", type=int, help="Number of threads")
    p.add_argument("chunk_size", type=int, help="Chunk size")
    ns = p.parse_args()
    return Args(ns.path, ns.threads, ns.chunk_size)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    numbers = read_ints(args.path)
    result = sum_primes_parallel(numbers, args.threads, args.chunk_size)
    print(f"Final Result: {result}")


if __name__ == "__main__":
    main()
