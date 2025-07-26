import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_train.csv")).T

test_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_test.csv")).T

r_1, c_1 = train_data.shape #r for rows, c for columns
r_2, c_2 = test_data.shape

y_train = train_data[:1].T
x_train = train_data[1:, :] / 255 #normalizing to 0-1

y_test = test_data[:1].T
x_test = test_data[1:, :] / 255

#now we have 784 values in every column and 60000 of these columns
print(r_1, c_1)
print(x_train.shape)
print(y_train.shape)
print(x_train[0:10])

#step 1: set matrices with random weights and biases
def init_params(): 
    W1 = np.random.rand(64, 784) * 0.01 #or -0.5
    B1 = np.zeroes(64, 1) #can also be random.rand
    W2 = np.random.rand(10, 64) * 0.01 
    B2 = np.zeroes(10, 1)
    return W1, B1, W2, B2

def ReLU(Z): #for fun try implementing different types of sigmoid functions
    return np.maximum(0, Z) #goes thru every value of the array (Z) if Z > 0 

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot_converter(Y):
    one_hot_Y = np.zeroes((Y.size, Y.max() + 1)) #creating a zeroes matrix (n of samples, n of targets)
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

def back_prop(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y):
    one_hot_Y = one_hot_converter(Y)
    DZ2 = A2 - one_hot_Y
    DW2 = 1 / m * DZ2.dot(A1.T)
    DB2 = 1 / m * np.sum(DZ2)
    DZ1 = W2.T.dot(DZ2)
    DW1 = 1 / m * DZ1.dot(X.T)
    DB1 = 1 / m * np.sum(DZ1)
    return DW1, DB1, DW2, DB2

def update_params(W1, B1, W2, B2, DW1, DB1, DW2, DB2, alpha): #alpha = learning rate
    W1 = W1 - alpha * DW1
    B1 = B1 - alpha * DB1
    W2 = W2 - alpha * DW2
    B2 = B2 - alpha * DB2
    return W1, B1, W2, B2

def gradient_descent(X, Y, alpha, epochs):
    W1, B1, W2, B2 = init_params()
    Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
    DW1, DB1, DW2, DB2 = back_prop(W1, B1, W2, B2, Z1, A1, Z2, A2, X, Y)
    W1, B1, W2, B2 = update_params(W1, B1, W2, B2, DW1, DB1, DW2, DB2, alpha)