import socket
from functools import reduce
from typing import List, Optional, Callable


# Step 1: Define reusable functional programming utilities
def square(x: int) -> int:
    """Square a number."""
    return x * x


def is_even(x: int) -> bool:
    """Check if a number is even."""
    return x % 2 == 0


def add(x: int, y: int) -> int:
    """Sum two numbers."""
    return x + y


# Step 2: File reading utility
def read_numbers_from_file(file_path: str) -> List[int]:
    """Read integers from a file."""
    try:
        with open(file_path, 'r') as file:
            return [int(line.strip()) for line in file.readlines()]
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        raise ValueError("File contains invalid data. Ensure all lines are integers.")


# Step 3: Data processing utility
def process_numbers(numbers: List[int], 
                    mapper: Callable[[int], int], 
                    filterer: Callable[[int], bool], 
                    reducer: Callable[[int, int], int], 
                    initial: int = 0) -> int:
    """
    Process numbers using map, filter, and reduce.

    :param numbers: List of numbers to process.
    :param mapper: Function to transform each number.
    :param filterer: Function to filter numbers.
    :param reducer: Function to reduce numbers.
    :param initial: Initial value for the reducer.
    :return: Final reduced value.
    """
    mapped = map(mapper, numbers)
    filtered = filter(filterer, mapped)
    return reduce(reducer, filtered, initial)


# Step 4: TCP communication utility
def send_message_via_tcp(message: str, host: str, port: int) -> None:
    """Send a message to a TCP server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))
            client_socket.sendall(message.encode('utf-8'))
            print(f"Message sent to {host}:{port}")
    except ConnectionError as e:
        raise ConnectionError(f"Unable to connect to {host}:{port}. {str(e)}")


# Step 5: Main function with reusable components
def main(file_path: str, tcp_host: str, tcp_port: int) -> None:
    """Main execution function."""
    try:
        # Step 1: Read numbers from the file
        numbers = read_numbers_from_file(file_path)

        # Step 2: Process the numbers
        result = process_numbers(numbers, square, is_even, add)

        # Step 3: Send the result via TCP
        send_message_via_tcp(f"Processed result: {result}", tcp_host, tcp_port)

        # Step 4: Display the result
        print(f"Result: {result}")

    except Exception as e:
        print(f"Error: {str(e)}")


# Configurations (adjust as needed)
if __name__ == "__main__":
    file_path = "numbers.txt"  # Replace with your file path
    tcp_host = "127.0.0.1"    # Replace with the server's IP
    tcp_port = 8080           # Replace with the server's port

    main(file_path, tcp_host, tcp_port)
