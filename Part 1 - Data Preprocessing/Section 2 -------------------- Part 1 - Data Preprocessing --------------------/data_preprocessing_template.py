# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values #:-1 means all columns beside the last one
Y = dataset.iloc[:, 3].values # we take the last column

#splitting the dataset into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Features scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)"""
