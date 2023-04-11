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
    data.append(d)

print(data[0])
print(data[1])
print("length: " + str(len(data)))



