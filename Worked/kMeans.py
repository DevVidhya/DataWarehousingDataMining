# Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load the train and test datasets to create two DataFrames
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#checking presence of missing values
#print(train.isna().sum())
#print(test.isna().sum())

# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set
test.fillna(test.mean(), inplace=True)

#see the survival count of passengers with respect to the features
#train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#train.info()

train = train.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)
test = test.drop(['Name','Ticket', 'Cabin','Embarked'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(train['Sex'])
labelEncoder.fit(test['Sex'])
train['Sex'] = labelEncoder.transform(train['Sex'])
test['Sex'] = labelEncoder.transform(test['Sex'])

# Let's investigate if you have non-numeric data left
#train.info()
#test.info()

X = np.array(train.drop(['Survived'], 1).astype(float))
y = np.array(train['Survived'])

#print(X.shape)
#print(y)

#KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=600,n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',random_state=None, tol=0.0001, verbose=0)
kmeans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
#    print(prediction[0],y[i])
    if (prediction[0] == y[i]):
        correct += 1

#print(correct)
#print(len(X))
print("Accuracy:")
print(float(correct)/float(len(X)))

#Result
#$ python kMeans.py
#Accuracy:
#0.626262626263




