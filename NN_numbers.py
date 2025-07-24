import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_train.csv")).T

test_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_test.csv")).T

r_1, c_1 = train_data.shape #r for rows, c for columns
r_2, c_2 = test_data.shape

y_train = train_data[:1]
x_train = train_data[:, 1:]

y_test = test_data[:1]
x_test = test_data[:, 1:]

#now we have 784 values in every column and 60000 of these columns
print(r_1, c_1)
print(y_train[0:10])
print(x_train[1])

#step 1: set matrices with random weights and biases
def init_params(): 
    W1 = np.random.rand(64, 784) * 0.01
    b1 = np.zeroes(64, 1)
    W2 = np.random.rand(10, 64) * 0.01
    b2 = np.zeroes(10, 1)
    return W1, b1, W2, b2

def ReLU(Z): #for fun try implementing different types of sigmoid functions
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