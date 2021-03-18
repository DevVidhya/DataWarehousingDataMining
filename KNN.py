### General import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import datetime
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.ion()

mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['figure.figsize'] = (8, 4)
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14

# Load and sort the dataframe.

df = pd.read_pickle('home_dat_20160918_20170604.pkl')
df.set_index('last_communication_time', inplace=True)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df.head()

# Pick out our time series object
# and fix it to a 5-min sampling period

y = df.available_bikes
y.index.name = 'time'
y = y.resample('5T').last()
y.head()

#Build our matrices

window = 5
num_samples = 8
X_mat = []
y_mat = []
for i in range(num_samples):
    # Slice a window of features
    X_mat.append(y.iloc[middle - i - window:middle - i].values)
    y_mat.append(y.iloc[middle - i:middle - i + 1].values)

X_mat = np.vstack(X_mat)
y_mat = np.concatenate(y_mat)

assert X_mat.shape == (num_samples, window)
assert len(y_mat) == num_samples

lr = LinearRegression(fit_intercept=False)
lr = lr.fit(X_mat, y_mat)
y_pred = lr.predict(X_mat)


