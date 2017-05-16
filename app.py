import network
import collect

training_data, validation_data, test_data = collect.load_mnist()

net = network.NeuralNetwork([784, 30, 10])
net.fit(training_data, validation_data)
