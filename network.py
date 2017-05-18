import os
import numpy as np
import random

# function sigmoid
# To calculate the activation value of the sigmoid neuron using the formula 
#  (1/1 + e^(wx+b))
# Attributes
# z- weigted sum of the inputs
# 
# Return
# an activation value in the range 0 to 1 

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# function sigmoid
# To calculate the activation value of the sigmoid neuron using the formula 
#  (1/1 + e^(wx+b))
# Attributes
# z- weigted sum of the inputs
# 
# Return
# an activation value in the range 0 to 1 

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# class NeuralNetwork 
# It creates a class of Neural Network for handling the artificial neural network operation
# 
class NeuralNetwork(object):
# Constructor to initialize the variable values.
#  Arguments:
# size : sizes represents the number of signmoid neurons in each layers of the 
# neural network  
# learning_rate: Takes a default value of 1 if not passed. Represents the relation
# between the change in weights and bias and the the cost function
# mini_batch_size: Size of subgroups obtianed from training set. Default values is 16
# epochs : Number of itreations to be performed. Takes a default value of 


    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16,
                 epochs=10):

        self.sizes = sizes
        self.num_layers = len(sizes)

        #  As no weights enter the input layer and hence self.weights[0] 
        # is redundant. Thus we set is as 0. For the rest we choose randomly
        # from tuples from 1 to the end and from first to penultiate values
        self.weights = [np.array([0])] + [np.random.randn(y, x) for y, x in
                                          zip(sizes[1:], sizes[:-1])]

        # As no bias is assosciated with first layer and for the remaining we 
        # choose random value of bias for each simoid neuron
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer.Thus we set it as zero for the first layer
        # and the remaining we set accordingly
        self._zs = [np.zeros(bias.shape) for bias in self.biases]

        # The activationsfor the first lyer is the terminal input.
        # It represents the output of every sigmoid layer
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.eta = learning_rate

    # function fit
    # To fit the neural network on the training data.
    #  
    # Attributes
    # training_data-The data using which teh neural network is trained
    # validation_data - The data using which the network is tested.It is set to
    # to none if not passed
    # 
    # Return
    # an activation value in the range 0 to 1 
    def fit(self, training_data, validation_data=None):

    # We iterate for every epoch for trianing the neural network 
        for epoch in range(self.epochs):
    # We shuffle the training data before splitting into mini batch
            random.shuffle(training_data)
    # Creating mini batches here
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]
    #  Iterating over each mini batch
            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    self.calc_forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self.calc_back_prop(y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.eta / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, delta_nabla_w)]
                self.biases = [
                    b - (self.eta / self.mini_batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]

        # If a validation data is given we calculate the accuracy.This is done for each
        # epoch
            if validation_data:
                accuracy = self.calc_accuracy(validation_data) / 100.0
                print("Epoch {0}, accuracy {1} %.".format(epoch + 1, accuracy))
            else:
                print("Processed epoch {0}.".format(epoch))

    # function calc_accuracuy
    # To caluclate home many images we accuarately detected
    #  
    # Attributes
    # validation_data - The data using which the network is tested.It is set to
    # to none if not passed
    # 
    # Return
    # The number of images correctly predicted

    def calc_accuracy(self, validation_data):
        # Identifying the number of correctly detected images
        validation_results = [(self.predict_image(x) == y) for x, y in validation_data]
        return sum(result for result in validation_results)

    # function predict_image
    # To predict what image the given set of features corresponds to
    #  
    # Attributes
    # features - The fetures corresponding to a given column
    # 
    # Return
    # The image which has the maximum probability for the given set of features
    def predict_image(self, features):
        self.calc_forward_prop(features)
        return np.argmax(self._activations[-1])

    # function calc_forward_prop
    # To calculate the z using the formula z= wx+b and activation using 
    # (1/1+e ^ (-z))
    #  
    # Attributes
    # x - The fetures corresponding to a given column
    # 
    # Return
    # None
    def calc_forward_prop(self, x):
    #  For teh first layer teh input is considered as activations
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])

    # function calc_back_prop
    # To calculate the cost function which is the function of w and b and update
    # the weight and bias accordingly to train the neural networrk for better efficency 
    #  
    # Attributes
    # y - The actual output of the given dataset 
    # 
    # Return
    # None
    def calc_back_prop(self, y):

        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
#    error value is calculated based on the predictvalue and the actual value
        error = (self._activations[-1] - y) * sigmoid_prime(self._zs[-1])
        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                sigmoid_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w