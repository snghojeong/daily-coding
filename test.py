import socket
from functools import reduce

# Step 1: Define a function to square a number
def square(x):
    return x * x

# Step 2: Define a function to filter even numbers
def is_even(x):
    return x % 2 == 0

# Step 3: Define a function to sum two numbers
def add(x, y):
    return x + y

# Step 4: Read numbers from a file and process them
def process_numbers_from_file(file_path):
    try:
        # Read numbers from the file
        with open(file_path, 'r') as file:
            numbers = [int(line.strip()) for line in file.readlines()]

        # Apply functional programming concepts
        squared_numbers = map(square, numbers)  # Map: Apply the `square` function to each number
        even_numbers = filter(is_even, squared_numbers)  # Filter: Keep only the even numbers
        total_sum = reduce(add, even_numbers, 0)  # Reduce: Compute the sum of the remaining numbers

        return total_sum
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError:
        print("Error: File contains non-numeric data.")
        return None

# Step 5: Send the result via TCP
def send_result_via_tcp(result, host, port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, port))
            message = f"Processed result: {result}"
            client_socket.sendall(message.encode('utf-8'))
            print(f"Sent result to {host}:{port}")
    except ConnectionError:
        print(f"Error: Could not connect to {host}:{port}")

# File path to the text file containing numbers
file_path = "numbers.txt"  # Replace with your file path

# TCP server details
tcp_host = "127.0.0.1"  # Replace with your server's IP
tcp_port = 8080         # Replace with your server's port

# Process the numbers and send the result via TCP
result = process_numbers_from_file(file_path)
if result is not None:
    print(f"Result: {result}")
    send_result_via_tcp(result, tcp_host, tcp_port)
