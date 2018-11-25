#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import clone, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils import plot_classification_dataset, plot_2d_decisionboundary


# One vs. Rest classifier
class OneVsRestClassifier(ClassifierMixin):
	def __init__(self, model, n_classes):
		self.n_classes = n_classes
		self.models = [clone(model) for _ in range(self.n_classes)]

	def fit(self, X, y):
		# Fit each classifier
		for i in range(self.n_classes):
			"""
			Here is one possible way shown to go over each model and train it.
			The label needs to be set so that the three classes are separated
			into one label for class 0 and the other two labels for class 1. WAS DENN NUN? DER ZETTEL SAGT ANDERS HERUM!!11
			Which label is put into class 0 needs to change every iteration.
			"""
			y_data = self.arrangeYInputForClass(y, i)
			self.models[i].fit(X, y_data)

	def arrangeYInputForClass(self, y, class_i):
		y_data = np.zeros(y.shape,dtype=np.int16)
		for j in range(y.shape[0]):
			if y[j] == class_i:
				y_data[j] = 1
		return y_data;

	def predict(self, X):
		# Compute predictions (probabilities) of each classifier
		predictions = np.zeros((self.n_classes, X.shape[0]))
		for i in range(self.n_classes):
			predictions[i] = self.models[i].predict(X);

		# Prediction with highest probability becomes the final prediction
		pred = np.zeros(X.shape[0],)
		for i in range(X.shape[0]):
			pred_class = 0
			_max = predictions[0][i]
			for j in range(self.n_classes):
				if predictions[j][i] > _max:
					pred_class = j
					_max = predictions[j][i]

			pred[i] = pred_class

		return pred

	def score(self, X, y):
		return accuracy_score(y, self.predict(X))


if __name__ == "__main__":

	# Load data set
	data = np.load('data5_1.npz')
	X, y = data['X'], data['y']

	# Split data into train and test set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

	# Train One vs. Rest classifierp
	model = OneVsRestClassifier(model=LogisticRegression(solver="liblinear", multi_class="auto"), n_classes=3)
	model.fit(X_train, y_train)

	# Evaluate classifiers
	# Accessing an individual classifier's score is done by model.models[i].score
	# For the OneVsRestClassifier use model.score

	# Plot classifiers
	print("OneVsRest Train-Score: {0}".format(model.score(X_train, y_train)))
	print("OneVsRest Test-Score: {0}".format(model.score(X_test, y_test)))
	plot_2d_decisionboundary(model, X, y);


	for i in range(model.n_classes):
		print("OneVsRest Train-Error: {0}".format(1 - model.models[i].score(X_train, model.arrangeYInputForClass(y_train, i))))
		print("OneVsRest Test-Error: {0}".format(1 - model.models[i].score(X_test, model.arrangeYInputForClass(y_test, i))))
		plot_2d_decisionboundary(model.models[i], X, model.arrangeYInputForClass(y, i))

#overall
#OneVsRest Train-Score: 0.9952380952380953
#OneVsRest Test-Score: 0.9777777777777777
#class 0
#OneVsRest Train-Error: 0.004761904761904745
#OneVsRest Test-Error: 0.0
#class 1
#OneVsRest Train-Error: 0.0
#OneVsRest Test-Error: 0.022222222222222254
#class 2
#OneVsRest Train-Error: 0.0
#OneVsRest Test-Error: 0.011111111111111072

	# You can iterate over models using model.models[i]
	# For visualisation the input y should be changed according to the
	# class separation you did the training for that classifier with

	plt.show()
