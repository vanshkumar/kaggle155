import sklearn 
import numpy as np
import os 
import dataIO as data
from sklearn import linear_model

def loadModelOut(filename):	
	return np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True)[1]

# load actual data properly 
# trainingDataFiles = ['sample_solution.txt', 'sample_solution_2.txt']
# testDataFiles = ['sample_solution.txt', 'sample_solution_2.txt']

# yTraining = data.loadTraining()['ylabels']
# xTraining = np.zeros((len(trainingDataFiles), len(yTraining)))
# xTraining = np.transpose([loadModelOut(model) for model in trainingDataFiles])

# xTest = np.zeros((len(testDataFiles), len(yTraining)))
# xTest = np.transpose([loadModelOut(model) for model in testDataFiles])

# bs data for now
xTraining = [[-1, -1], [-2, -1], [1, 1], [2, 1]]
yTraining = [1, 1, 2, 2]
xTest = [[-1, -1], [-2, -1], [1, 1], [2, 1]]


clf = linear_model.SGDClassifier()
clf.fit(xTraining, yTraining)
yTest = clf.predict(xTest)



f = open('blended_solution.csv', 'w+')
f.write('Id,Prediction\n')
for i in range(len(yTest)):
    f.write(str(i+1) + ',' + str(yTest[i]) + '\n')
f.close()