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

def ReLU(Z):
    return np.maximum(0, Z) #goes thru every value of the array (Z) if Z > 0 

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def back_prop(): #I'll need to study more theory before i can confidently do this
    pass