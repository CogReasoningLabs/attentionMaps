"""Bounded I/O concurrency without changing source order or sampling seeds."""

from concurrent.futures import ThreadPoolExecutor

MAX_READ_WORKERS = 16


def ordered_parallel_reads(function, items, workers):
    """Read independent items concurrently and return results in input order."""
    if type(workers) is not int or not 1 <= workers <= MAX_READ_WORKERS:
        raise ValueError(f"Read workers must be between 1 and {MAX_READ_WORKERS}")
    if workers == 1 or len(items) < 2:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(workers, len(items)), thread_name_prefix="dataset-read") as pool:
        return list(pool.map(function, items))
