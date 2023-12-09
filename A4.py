# Lucian Tranc 1045249
import numpy as np
import time

# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# sum sqaured difference function
def sum_squared_difference(output, expected_result):
    return np.sum((expected_result - output) ** 2) / len(output)

# derivative of sum sqaured difference function
def sum_squared_difference_derivative(output, expected_result):
    return (2/len(output)) * (output - expected_result)

# turns the integer value of the expected result into an array where the 
# index value of the expected result is 1, and all others are 0
def expected_result_array(expected_result):
    expected_array = np.zeros(10)
    expected_array[expected_result] = 1
    return expected_array

# forward propogation function
def forward_propagation(input, nodesPerLayer, weights, biases):

    # initialize empty set of nodes
    output = []
    for nodeCount in nodesPerLayer:
        output.append(np.zeros(nodeCount, float))
    
    # set first layer to input
    output[0] = input

    # set the value of each node based on the preivous layer's values, the weights and the biases.
    for layerIndex in range(1, len(nodesPerLayer)):
        for nodeIndex in range(0, len(weights[layerIndex - 1])):
            output[layerIndex][nodeIndex] = sigmoid(np.sum(np.multiply(output[layerIndex - 1], weights[layerIndex - 1][nodeIndex])) + biases[layerIndex-1][nodeIndex])
    return output

# backward propagation fnction
def backward_propagation(output, expected_result, nodesPerLayer, weights, biases):

    # initialize empty set of errors
    errors = []
    for nodeCount in nodesPerLayer:
        errors.append(np.zeros(nodeCount, float))
    
    # calculate the error for the output layer
    errors[-1] =  sigmoid_derivative(np.dot(weights[-1], output[-2]) + biases[-1]) * sum_squared_difference_derivative(output[-1], expected_result)
    
    # calculate errors
    for layerIndex in range(len(nodesPerLayer) - 2, 0, -1):
        errors[layerIndex] = sigmoid_derivative(np.dot(weights[layerIndex - 1], output[layerIndex - 1]) + biases[layerIndex - 1]) * np.dot(errors[layerIndex + 1], weights[layerIndex])
    
    # gradient calculations
    gradient_weights = []
    gradient_biases = []
    for layerIndex in range(len(nodesPerLayer) - 1):
        gradient_weights.append(np.outer(errors[layerIndex + 1], output[layerIndex]))
        gradient_biases.append(errors[layerIndex + 1])
    
    # return the updated weights and biases
    return gradient_weights, gradient_biases


data = [] # initialize data array
file = open('data/mnist_train.csv', 'r') # open file object
file.readline() # skip the first line
for d in file:
    d = d[:-1]
    split = d.split(",")
    data.append([int(split[0]), np.asarray(split[1:], float)/255])

testing_data = [] # initialize data array
file = open('data/mnist_test.csv', 'r') # open file object
file.readline() # skip the first line
for d in file:
    d = d[:-1]
    split = d.split(",")
    testing_data.append([int(split[0]), np.asarray(split[1:], float)/255])

# 88.19% accuracy after 18 epochs
# nodesPerLayer = [28*28, 512, 10]
# learningRate = 1

# 97.54% accuracy after 18 epochs
# nodesPerLayer = [28*28, 512, 10]
# learningRate = 0.7

# 97.14% accuracy after 14 epochs
# nodesPerLayer = [28*28, 200, 10]
# learningRate = 0.5

# 96.54% accuracy after 14 epochs
# nodesPerLayer = [28*28, 50, 10]
# learningRate = 0.5

# 96.25% accuracy after 9 epochs
# nodesPerLayer = [28*28, 50, 10]
# learningRate = 1

# 95.21% accuracy after 10 epochs - Took forever to train
# nodesPerLayer = [28*28, 200, 80, 10]
# learningRate = 0.1


# this array specifies the number of nodes on each layer.
nodesPerLayer = [28*28, 20, 10]

learningRate = 1

epochs = 20

weights = []
biases = []
for i in range(1, len(nodesPerLayer)):
    weights.append(np.random.rand(nodesPerLayer[i], nodesPerLayer[i-1])*2 - 1)
for i in range(1, len(nodesPerLayer)):
    biases.append(np.random.randn(nodesPerLayer[i]))

weight_delta = []
bias_delta = []
for i in range(1, len(nodesPerLayer)):
    weight_delta.append(np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])))
for i in range(1, len(nodesPerLayer)):
    bias_delta.append(np.zeros(nodesPerLayer[i]))



# loop for epochs
for e in range(0, epochs):
    print("Epoch: " + str(e + 1))
    start = time.time()
    # shuffle the data
    np.random.shuffle(data)
    for d in data:
        expected = expected_result_array(d[0])
        output = forward_propagation(d[1], nodesPerLayer, weights, biases)
        # get the gradiant of the weights and biases
        weight_gradient, bias_gradient = backward_propagation(output, expected, nodesPerLayer, weights, biases)
        # apply it to the wieghts and biases
        for i in range(0, len(weights)):
            weights[i] -= weight_gradient[i] * learningRate

        for i in range(0, len(biases)):
            biases[i] -= bias_gradient[i] * learningRate

    # testing code

    correct = 0
    error = 0

    for d in data:
        expected = expected_result_array(d[0])
        output = forward_propagation(d[1], nodesPerLayer, weights, biases)
        error_i = sum_squared_difference(output[len(nodesPerLayer)-1], expected)
        error = error + error_i
        index_max = np.argmax(output[len(nodesPerLayer)-1])
        if (d[0] == index_max):
            correct += 1

    print("training: " + str(correct) + "/" + str(len(data)))
    print("training error: " + str(error/len(data)))

    correct = 0
    error = 0

    for d in testing_data:
        expected = expected_result_array(d[0])
        output = forward_propagation(d[1], nodesPerLayer, weights, biases)
        error_i = sum_squared_difference(output[len(nodesPerLayer)-1], expected)
        error = error + error_i
        index_max = np.argmax(output[len(nodesPerLayer)-1])
        if (d[0] == index_max):
            correct += 1

    print("testing: " + str(correct) + "/" + str(len(testing_data)))
    print("testing error: " + str(error/len(testing_data)))
    end = time.time()
    print("time: " + str(end - start))
    print("")
