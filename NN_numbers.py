import numpy as np
import pandas as pd

train_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_train.csv")).T

test_data = np.array(pd.read_csv("/Users/amir/Downloads/CodeAmir/NeuralNets/MNIST as csv/mnist_test.csv")).T

r_1, c_1 = train_data.shape #r for rows, c for columns
r_2, c_2 = test_data.shape

y_train = train_data[0]
x_train = train_data[1:r_1]

y_test = test_data[0]
x_test = test_data[1:r_2]

print(r_1,c_1, r_2, c_2)

#the entire row, 0th column