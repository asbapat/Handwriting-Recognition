import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_images(training_set):
    """ Return a list containing the images from the MNIST data
    set. Each image is represented as a 2-d numpy array."""
    flattened_images = training_set[0]

    return [np.reshape(f, (-1, 28)) for f in flattened_images]

def plot_images_together(images):
    """ Plot a single image containing all six MNIST images, one after
    the other.  Note that we crop the sides of the images so that they
    appear reasonably close together."""
    fig = plt.figure()
    images = [image[:, 3:25] for image in images]
    image = np.concatenate(images, axis=1)
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()   

# function plotDigit
# To plot a particular digit in the Dataset and show it as an image
# Attributes
# image- The 28 x28 pixel image in the dataset
# 
# Return
# dataset as a list    

def plotDigit(image):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

# function plot_box_images
# To plot a matrix of digits with information present in the dataset
# Attributes
# images- A set of 28 x28 pixel images to be plotted in the form of a matrix
# 
# Return
# None      


def plot_box_images(images):
    fig = plt.figure()
    # We crop the images so that they appear reasonably close together.
    images = [image[3:25, 3:25] for image in images]
    # we tahe the range fro 1 to 9 to plot 9 x 9 matrix of images
    for x in range(1,10):
        for y in range(1,10):
        	# Subplots are created to plot the image
            ax = fig.add_subplot(10, 10, 10*y+x)
            ax.matshow(images[10*y+x], cmap = matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()     