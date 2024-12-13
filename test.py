from rx import from_list, operators as ops
from rx.scheduler import ThreadPoolScheduler
import socket
import threading


# Step 1: Define reusable functions for data transformations
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
def read_numbers_from_file(file_path: str):
    """Read integers from a file and return an observable."""
    try:
        with open(file_path, 'r') as file:
            numbers = [int(line.strip()) for line in file.readlines()]
        return from_list(numbers)  # Create an observable from the list
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except ValueError:
        raise ValueError("File contains invalid data. Ensure all lines are integers.")


# Step 3: TCP communication utility
def send_message_via_tcp(message: str, host: str, port: int) -> None:
    """Send a message to a TCP server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))
            client_socket.sendall(message.encode('utf-8'))
            print(f"Message sent to {host}:{port}")
    except ConnectionError as e:
        raise ConnectionError(f"Unable to connect to {host}:{port}. {str(e)}")


# Step 4: Main reactive processing function
def reactive_process(file_path: str, tcp_host: str, tcp_port: int):
    """Process the numbers from a file reactively and send results over TCP."""
    # Use a thread pool scheduler for asynchronous operations
    thread_scheduler = ThreadPoolScheduler(threading.active_count())

    try:
        # Reactive pipeline
        read_numbers_from_file(file_path).pipe(
            ops.map(square),                # Map: Apply square transformation
            ops.filter(is_even),            # Filter: Keep even numbers
            ops.reduce(add, seed=0)         # Reduce: Sum up the numbers
        ).subscribe(
            on_next=lambda result: (
                send_message_via_tcp(f"Processed result: {result}", tcp_host, tcp_port),
                print(f"Result: {result}")
            ),
            on_error=lambda e: print(f"Error: {str(e)}"),
            scheduler=thread_scheduler
        )
    except Exception as e:
        print(f"Error: {str(e)}")


# Configurations (adjust as needed)
if __name__ == "__main__":
    file_path = "numbers.txt"  # Replace with your file path
    tcp_host = "127.0.0.1"    # Replace with the server's IP
    tcp_port = 8080           # Replace with the server's port

    reactive_process(file_path, tcp_host, tcp_port)
