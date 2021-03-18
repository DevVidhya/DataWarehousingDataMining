import pandas as pd

df = pd.read_excel("bank.xlsx")

df['target'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

df.isnull().mean().sort_values(ascending=False)*100

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
