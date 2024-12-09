from functools import reduce

# Define a list of numbers
numbers = [1, 2, 3, 4, 5]

# Step 1: Define a function to square a number
def square(x):
    return x * x

# Step 2: Define a function to filter even numbers
def is_even(x):
    return x % 2 == 0

# Step 3: Define a function to sum two numbers
def add(x, y):
    return x + y

# Apply functional programming concepts
# Map: Apply the `square` function to each number
squared_numbers = map(square, numbers)

# Filter: Keep only the even numbers
even_numbers = filter(is_even, squared_numbers)

# Reduce: Compute the sum of the remaining numbers
total_sum = reduce(add, even_numbers)

# Print the result
print(f"Result: {total_sum}")
