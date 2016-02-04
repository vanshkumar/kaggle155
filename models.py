import sklearn 
import numpy as np
import os 
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import dataIO as data

def gaussianNBayes(x, y, x_predict):
    '''Gaussian Naive Bayes algorithm.  Input is xlabels, ylabels, and the data
    you want to predict on.  Returns an array of predictions'''
    gnb = GaussianNB()
    y_pred = gnb.fit(x, y).predict(x_predict)
    return y_pred 

def ridgeRegression(x, y, x_predict, in_alpha):
    '''Ridge regresssion model.  in_alpha is used as a regularization parameter.
    returns the predicted values for the x_predict array'''
    clf = linear_model.Ridge (alpha = in_alpha)
    return (clf.fit(x, y)).predict(x_predict)

def randomForest(x, y, x_predict):
    # There are like 20+ parameters that can be tuned for this model :'(
    rf = RandomForestClassifier()
    return rf.fit(x, y).predict(x_predict)


def exampleUsage():
    '''This is just here to show how to run the models'''
    training_data = data.loadTraining()
    out = gaussianNBayes(training_data['xlabels'], training_data['ylabels'], \
                         training_data['xlabels'])
    out1 = ridgeRegression(training_data['xlabels'], training_data['ylabels'], \
                         training_data['xlabels'], 3)
    out2 = randomForest(training_data['xlabels'], training_data['ylabels'], \
                         training_data['xlabels'])
