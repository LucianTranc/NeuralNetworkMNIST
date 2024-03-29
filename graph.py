import pandas as pd
import matplotlib.pyplot as plt
import argparse
import datetime
import re

# Create an argument parser
parser = argparse.ArgumentParser(description='Plot the testing accuracy for each learning rate')
parser.add_argument('csv_file', type=str, help='The CSV file to process')

# Parse the command line arguments
args = parser.parse_args()

# Extract the number of hidden nodes from the filename
hidden_nodes = re.search(r'test_results_((\d+_)*\d+)', args.csv_file).group(1)
hidden_nodes_comma = hidden_nodes.replace('_', ', ')

# Read the CSV file into a DataFrame
df = pd.read_csv(args.csv_file)

# Define the epochs
epochs = range(1, int(df.shape[1] - 2) + 1)

# Group the DataFrame by the learning rate
grouped = df.groupby('Learning Rate')

# Create a new figure
plt.figure(figsize=(10, 6))

# Iterate over each group in the DataFrame
for name, group in grouped:
    # Extract the testing accuracies for each epoch
    testing_accuracies = group[[f'Testing Accuracy (Epoch {epoch})' for epoch in epochs]].mean()

    # Calculate the maximum accuracy
    max_accuracy = testing_accuracies.max()

    time = group['Training Time']

    training_time = datetime.timedelta(seconds=int(time.iloc[0]))

    # Plot the testing accuracies versus the epochs
    plt.plot(epochs, testing_accuracies, marker='o', linestyle='-', label=f'Learning Rate: {name}, Max Accuracy: {max_accuracy:.4f}, Training Time: {training_time}')

# Add labels, a title, and a legend
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.title(f'Testing Accuracy for {hidden_nodes_comma} Hidden Nodes')
plt.legend()

# Set the x-axis ticks to be whole numbers
plt.xticks(epochs)

# Set the y-axis limits to be between 0 and 1
plt.ylim(0.2, 1)

# Save the plot
plt.savefig(f'graphs/Testing_Accuracy_{hidden_nodes}.png')

# Show the plot
plt.show()

