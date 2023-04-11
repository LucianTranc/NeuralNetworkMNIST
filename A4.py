import numpy as np

# neural network class. For now I will hardcode 2 hidden layers of size 16
# I need to represent all the nodes, grouped in layers
# I need to represent all the weights between each layer
# I need to represent the bias of each node in each layer
class NeuralNetwork:
    def __init__ (self, ):
        self.inputLayer = []



print("Lucian Tranc's MNIST Nerual Network")

# Overview of next steps:

#   Set up input layer
#       This will just be an array of length 784.
#       Create a function that loads in an mnist datapoint into the array

data = [] # initialize data array
file = open('data/mnist_train.csv', 'r') # open file object
file.readline() # skip the first line
for d in file:
    d = d[:-1]
    split = d.split(",")
    data.append([split[0], split[1:]])

print(data[0])
print("length data: " + str(len(data)))
print("length data[0]: " + str(len(data[0])))
print("length data[0][1]: " + str(len(data[0][1])))

# start by implementing using an iterative approach and then encapsulate it in a class

firstLayerCount = 16
secondLayerCount = 16

# each input node will need a wieght going to each node in the first layer

# 784 * 16 weights going from input layer to first hidden layer
# 16 * 16 weights going from the first to the second hidden layer
# 16 * 10 weights going from second hidden layer to the ouput layer

# initialize the arrays of weights to random values


for image in data:
    inputLayer = image[1]

inputLayer = []





