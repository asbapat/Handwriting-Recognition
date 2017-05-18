import os
import gzip
import cPickle
import wget

import numpy as np

# function plot_box_images
# To plot a matrix of digits with information present in the dataset
# Attributes
# images- A set of 28 x28 pixel images to be plotted in the form of a matrix
# 
# Return
# None 

def load_data():

    # os is a python library that is used to perform file explorer operations
    #os like create directory, join directory
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))
        # we are downloading the dataset using thw wget library 
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz', out='data')
    # As the data will be gzip compressed it is opened using gzip.open and in read binary mode
    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
    # The below code is used to split the data automatically into testing and validation set
    training_data, validation_data, test_data = cPickle.load(data_file)
    data_file.close()

    # The following code reshapes the 28 x28 array into 784 coluns in single row
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]

    # The zip function is used to create tuples in python with first value astraining_inputs 
    # and second as testing inputs

    training_data_reshaped = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_results = validation_data[1]
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])

    return training_data_reshaped, validation_data, test_data ,training_data

# function vectorized_result
# To create an array of size 10 and set the index corresponsing to the class as 1
# Attributes
# y- represents the class. label number
# 
# Return
# e - an array of size 10 with a particular index set as 1 
def vectorized_result(y):
    # creates an array of size 10 and fills it with zero
    e = np.zeros((10, 1))
    # set the index corresponding to the label/ class value as 1 to indicate correct output
    e[y] = 1.0
    return e