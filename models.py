import sklearn 
import numpy as np
import os 
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import dataIO as data

def gaussianNBayes(x, y, x_predict):
    '''Gaussian Naive Bayes algorithm.  Input is xlabes, ylabels, and the data
    you want to predict on.  Returns and array of predictions'''
    gnb = GaussianNB()
    training_data = data.loadTraining()
    y_pred = gnb.fit(x, y).predict(x_predict)
    return y_pred 

def ridgeRegression(x, y, x_predict, in_alpha):
    '''Ridge regresssion model.  in_alpha is used as a regularization parameter.
    returns the predicted values for the x_predict array'''
    clf = linear_model.Ridge (alpha = in_alpha)
    return (clf.fit(x, y)).predict(x_predict)




def exampleUsage():
    '''This is just here to show how to run the models'''
    training_data = data.loadTraining()
    out = gaussianNBayes(training_data['xlabels'], training_data['ylabels'], \
                         training_data['xlabels'])
    out1 = ridgeRegression(training_data['xlabels'], training_data['ylabels'], \
                         training_data['xlabels'], 3)
