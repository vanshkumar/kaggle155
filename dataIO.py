import os 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

model_size = .7

def allTrain():
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
    xlabels = StandardScaler().fit_transform(xlabels)
    # pca = PCA(n_components=2*xlabels.shape[1]/3)
    # xlabels = pca.fit_transform(xlabels)
    return {'xlabels': xlabels, 'ylabels': ylabels}

def allTest():
    path = os.getcwd() + '/../WordSentiment'
    f = open(path + '/testing_data.txt', 'r')
    lines = f.readlines()
    lines = [x.split('|') for x in lines]
    words = lines[0]
    lines.remove(lines[0])
    lines = np.array([[float(x) for x in row] for row in lines])
    f.close()
    lines = StandardScaler().fit_transform(lines)
    # pca = PCA(n_components=2*lines.shape[1]/3)
    # lines = pca.fit_transform(lines)
    return {'xlabels': lines}


def loadTraining():
    all_train = allTrain()
    xlabels = all_train['xlabels']
    ylabels = all_train['ylabels']

    return {'xlabels': xlabels[:model_size*len(xlabels), :], 'ylabels': ylabels[:model_size*len(xlabels)]}

def loadTest():
    all_test = allTrain()
    xlabels = all_test['xlabels']
    ylabels = all_test['ylabels']

    return {'xlabels': xlabels[:model_size*len(xlabels), :], 'ylabels': ylabels[:model_size*len(xlabels)]}


def loadTrainingBlend():
    all_train = allTrain()
    xlabels = all_train['xlabels']
    ylabels = all_train['ylabels']

    return {'xlabels': xlabels[model_size*len(xlabels):, :], 'ylabels': ylabels[model_size*len(xlabels):]}

def loadTestBlend():
    all_test = allTrain()
    xlabels = all_test['xlabels']
    ylabels = all_test['ylabels']

    return {'xlabels': xlabels[model_size*len(xlabels):, :], 'ylabels': ylabels[model_size*len(xlabels):]}