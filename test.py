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
        # Map: Apply the `square` function to each number
        squared_numbers = map(square, numbers)

        # Filter: Keep only the even numbers
        even_numbers = filter(is_even, squared_numbers)

        # Reduce: Compute the sum of the remaining numbers
        total_sum = reduce(add, even_numbers, 0)  # Start reduce with 0 for an empty list

        return total_sum
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError:
        print("Error: File contains non-numeric data.")
        return None

# File path to the text file containing numbers
file_path = "numbers.txt"  # Replace with your file path

# Process the numbers and print the result
result = process_numbers_from_file(file_path)
if result is not None:
    print(f"Result: {result}")
