import numpy as np
import random

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_error(output, expected_result):
    expected_array = np.zeros(10)
    expected_array[expected_result] = 1
    return np.sum(np.square(np.subtract(expected_array, output)))


def forward_propagation(input, nodesPerLayer, weights, biases):

    # initialize empty set of nodes
    nodes = []
    for nodeCount in nodesPerLayer:
        nodes.append(np.zeros(nodeCount, float))
    
    # set first layer to input
    nodes[0] = input

    # set the value of each node based on the preivous layer's values, the weights and the biases.
    for layerIndex in range(1, len(nodesPerLayer)):
        for nodeIndex in range(0, len(weights[layerIndex - 1])):
            nodes[layerIndex][nodeIndex] = sigmoid(np.sum(np.multiply(nodes[layerIndex - 1], weights[layerIndex - 1][nodeIndex])) + biases[layerIndex-1][nodeIndex])
    return nodes


    
print("Lucian Tranc's MNIST Nerual Network")

data = [] # initialize data array
file = open('data/mnist_train.csv', 'r') # open file object
file.readline() # skip the first line
for d in file:
    d = d[:-1]
    split = d.split(",")
    data.append([int(split[0]), np.asarray(split[1:], float)/255])


# this array specifies the number of nodes on each layer.
nodesPerLayer = [28*28, 16, 16, 10]


weights = []

for i in range(1, len(nodesPerLayer)):
    weights.append(np.random.rand(nodesPerLayer[i], nodesPerLayer[i-1]) - 0.5)

print("Weights:")
print(weights)

biases = []

for i in range(1, len(nodesPerLayer)):
    biases.append(np.random.randn(nodesPerLayer[i]))

print("Biases:")
print(biases)

# to run a peice of data through the network, I need to multiply each value in the 
# input layer by the weight associated with the first hidden layer.
# I organized the arrays to have 16 arrays of 784 which means I can loop through each index

# weights: [Input to hidden layer 1: [layer 1 node 1 to input layer weights: [], ...], 
#           hidden layer 1 to hidden layer 2: [],
#           hidden layer 2 to ouput layer:[]]

# need to do vector multiplication between weights[0][node] and d

# At this point I have a list of nodes where the first layer is set to the input.
# I also have a sets of random weights and biases.
# I want to run a forward propogation. the forward propogation needs to keep the values
# of the nodes.

# gets the state of all the nodes

error = 0



loopLimiter = 0
for d in data:
    nodes = forward_propagation(d[1], nodesPerLayer, weights, biases)
    error = error + calculate_error(nodes[len(nodesPerLayer)-1], d[0])

    # calculate the error rate

print("total error: " + str(error))

# at this point we have the total error of the cost function

# we need to calculate the gradient vector of the cost function and apply it to the weights to nudge it in the right direction
# This can be thought of as deltaW (the vector storing the change we need for each weight)
# deltaW can be computed by multiplying (learning rate) x (output/activating of previous layer node) x (derivate of sigmoid(current node) x sum(weights x (error X derivative of sigmod(k))))




# i have an array of random weights and random biases based on an array that specifies how many nodes will be in each layer.

# technically, I have created a neural network. I could run the testing data through this network, and get an accuracy of 10%.
# I'm going to try this, and then compare it to a network I find online.

# the first step in running the neural network is to start with the data as the first layer
# each node in the second layer gets calculated by summing all the products of the weights and the values in each input node.





