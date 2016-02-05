import os 
import numpy as np


def loadTraining():
    # works if the word sentiment folder and kaggle folder are in the same dir
    path = os.getcwd() + '/../WordSentiment'
    f = open(path + '/training_data.txt', 'r')
    lines = f.readlines()
    lines = [x.split('|') for x in lines]
    words = lines[0]
    lines.remove(lines[0])
    ylabels = [x[-1].rstrip() for x in lines]
    xlabels = [x[:-1] for x in lines]
    # converts all the values to ints instead of strings
    xlabels = np.array([[float(x) for x in row] for row in xlabels])
    ylabels = np.array([float(x) for x in ylabels])
    f.close()
    return {'words': words, 'xlabels': xlabels, 'ylabels': ylabels}


def loadTest():
    path = os.getcwd() + '/../WordSentiment'
    f = open(path + '/testing_data.txt', 'r')
    lines = f.readlines()
    lines = [x.split('|') for x in lines]
    words = lines[0]
    lines.remove(lines[0])
    lines = np.array([[float(x) for x in row] for row in lines])
    f.close()
    return {'words': words, 'xlabels': lines}
