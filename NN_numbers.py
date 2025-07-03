import numpy as np
import pandas as pd

data = np.array(pd.read_csv(""))

r, c = data.shape #r for rows, c for columns

train_data = data[0:1000].T

