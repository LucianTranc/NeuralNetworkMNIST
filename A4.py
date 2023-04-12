import numpy as np
import random

#    nodes = forward_propogation(input, nodesPerLayer, weights, biases)

def forward_propogation(input, nodesPerLayer, weights, biases):

    # initialize empty set of nodes
    nodes = []
    for nodeCount in nodesPerLayer:
        nodes.append(np.zeros(nodeCount, float))
    
    nodes[0] = input

    nodeIndex = 0
    layerIndex = 1
    nodes[layerIndex][nodeIndex] = nodes[layerIndex-1] * weights[0]
    




    return 0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
print("Lucian Tranc's MNIST Nerual Network")

data = [] # initialize data array
file = open('data/mnist_train.csv', 'r') # open file object
file.readline() # skip the first line
for d in file:
    d = d[:-1]
    split = d.split(",")
    data.append([split[0], np.asarray(split[1:], float)])


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

# to run a peice of data through the network, I need to multiply each value in the 
# input layer by the weight associated with the first hidden layer.
# I organized the arrays to have 16 arrays of 784 which means I can loop through each index

# weights: [Input to hidden layer 1: [layer 1 node 1 to input layer weights: [], ...], 
#           hidden layer 1 to hidden layer 2: [],
#           hidden layer 2 to ouput layer:[]]

# need to do vector multiplication between weights[0][node] and d

print("WEIGHTS[0][0]")
print(weights[0][0])
print(len(weights[0][0]))

print("DATA[0][1]")
print(data[0][1])
print(len(data[0][1]))

for d in data:
    label = d[0]
    input = d[1]

    # At this point I have a list of nodes where the first layer is set to the input.
    # I also have a sets of random weights and biases.
    # I want to run a forward propogation. the forward propogation needs to keep the values
    # of the nodes.

    # gets the state of all the nodes
    nodes = forward_propogation(input, nodesPerLayer, weights, biases)





# i have an array of random weights and random biases based on an array that specifies how many nodes will be in each layer.

# technically, I have created a neural network. I could run the testing data through this network, and get an accuracy of 10%.
# I'm going to try this, and then compare it to a network I find online.

# the first step in running the neural network is to start with the data as the first layer
# each node in the second layer gets calculated by summing all the products of the weights and the values in each input node.





