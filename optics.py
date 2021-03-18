import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import gridspec 
from sklearn.cluster import OPTICS, cluster_optics_dbscan 
from sklearn.preprocessing import normalize, StandardScaler

X = pd.read_csv('Mall.csv') 
  
# Dropping irrelevant columns 
drop_features = ['CustomerID', 'Gender'] 
X = X.drop(drop_features, axis = 1) 
  
# Handling the missing values if any 
X.fillna(method ='ffill', inplace = True) 
  
X.head() 
