import numpy as np
import pandas as pd

train_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_train.csv"))

test_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_test.csv"))

r_1, c_1 = train_data.shape #r for rows, c for columns
r_2, c_2 = test_data.shape

y_train = train_data[:, 0]
x_train = train_data[:, 1:]

y_test = test_data[:, 0]
x_test = test_data[:, 1:]

def init_params(): #initializing weights and biases, weights are initally random but then optimized with backpropagation, biases are an array of zeroes, but get optimized later
    W1 = np.random.randn(784, 64) * 0.01
    b1 = np.zeroes(1,64)
    W2 = np.random.randn(64, 10) * 0.01
    b2 = np.zeroes(1,10)
    return W1, b1, W2, b2
