import statistics

def process_log_file(file_path):
    numbers = []

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into parts based on commas
            parts = line.split(',')
            if len(parts) >= 3 and 'python3' in parts[1]:
                try:
                    # Extract the number after 'python3' (which is the third part)
                    num = int(parts[2].strip())
                    numbers.append(num)
                except ValueError:
                    # Handle the case where the number is not an integer
                    print(f"Warning: Could not parse number in line: {line}")

    if numbers:
        # Calculate and print statistics
        min_value = min(numbers)
        max_value = max(numbers)
        median_value = statistics.median(numbers)
        avg_value = sum(numbers) / len(numbers)

        print(f"Min: {min_value}")
        print(f"Max: {max_value}")
        print(f"Median: {median_value}")
        print(f"Average: {avg_value:.2f}")
    else:
        print("No valid numbers found in the file.")

# Example usage:
file_path = 'gpu_python_memory_usage.log'  # Change this to the path of your file
process_log_file(file_path)

