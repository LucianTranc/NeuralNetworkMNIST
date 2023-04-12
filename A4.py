import numpy as np
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("Lucian Tranc's MNIST Nerual Network")

data = [] # initialize data array
file = open('data/mnist_train.csv', 'r') # open file object
file.readline() # skip the first line
for d in file:
    d = d[:-1]
    split = d.split(",")
    data.append([split[0], split[1:]])


# this array specifies the number of nodes on each layer.
nodesPerLayer = [28*28, 16, 16, 10]

weights = []

for i in range(1, len(nodesPerLayer)):
    weights.append(sigmoid(np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1])))

print("Weights:")
print(weights)

biases = []

for i in range(1, len(nodesPerLayer)):
    biases.append(sigmoid(np.random.randn(nodesPerLayer[i])))

print("Biases:")
print(biases)


# i have an array of random weights and random biases based on an array that specifies how many nodes will be in each layer.

# technically, I have created a neural network. I could run the testing data through this network, and get an accuracy of 10%.
# I'm going to try this, and then compare it to a network I find online.

# the first step in running the neural network is to start with the data as the first layer
# each node in the second layer gets calculated by summing all the products of the weights and the values in each input node.





