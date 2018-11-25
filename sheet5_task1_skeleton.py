#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from utils import plot_2d_decisionboundary


if __name__ == "__main__":

    # Arbitrary numbers that may be needed in multiple parts of the code
    # should be extracted and saved at a place that is easy to find
    number_of_iterations = 5

    # Load data set
    X, y = load_iris(True)
    # for sklearn 0.17 instead use:
    #X, y = load_iris().data, load_iris().target

    # Split into train and test set
    # A good idea may be to use 'random_state=<some number>' to get different splits
    # TODO

    # Use the first two dimensions to train a model
    # Document the error for the trained classifier
    # Plot the result
    # TODO

    # Now do the same for the last two dimensions
    # TODO

    # Use all dimensions
    # Plotting this will be difficult. You don't need to do that for this classifier
    # TODO o
