#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":

    # Load data set similar to task 2
    data = np.load('data5_2.npz')
    X, y = data['X'], data['y']

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Fit & evaluate knn
    model = KNeighborsClassifier(n_neighbors=5,algorithm='ball_tree')
    # See the footnotes for information on how to get the system's time stamp
    # Evaluate time needed for training a knn classifier
    now = time.time()
    model.fit(X_train, y_train)
    traintime = time.time() - now
    print("KNN-Train time: {0:.3f}s".format(traintime))

    # Evaluate time needed for predicting with a knn classifier
    # The actual result of the prediction is of no interest here
    now = time.time()
    model.predict(X_test)
    predicttime= time.time() - now
    print("KNN-Predict time: {0:.3f}s".format(predicttime))

    # Fit & evaluate logistic regression
    model = LogisticRegression(solver="liblinear")
    # Evaluate time needed for training a logistic regression classifier
    now = time.time()
    model.fit(X_train, y_train)
    traintime = time.time() - now
    print("LR-Train time: {0:.3f}s".format(traintime))

    # Evaluate time needed for predicting with a logistic regression classifier
    # The actual result of the prediction is of no interest here
    now = time.time()
    model.predict(X_test)
    predicttime= time.time() - now
    print("LR-Predict time: {0:.3f}s".format(predicttime))
