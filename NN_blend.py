import sklearn 
import numpy as np
import os 
import dataIO as data
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPClassifier


def loadModelOut(filename): 
    return np.loadtxt(os.getcwd() + '/../WordSentiment/' + filename, delimiter=',', skiprows=1, unpack=True)[1]

trainingDataFiles = ['ada_regress_train.csv', 
                     'gnb_class_train.csv', 
                     'knn_regress_train.csv', 
                     'log_regress_train.csv', 
                     'nn_class_train.csv',
                     'rf_regress_train.csv',
                     'rf_regress_train_2.csv',
                     'svm_regress_train.csv'
                     ]

testDataFiles = ['ada_regress_test.csv', 
                 'gnb_class_test.csv', 
                 'knn_regress_test.csv', 
                 'log_regress_test.csv', 
                 'nn_class_test.csv',
                 'rf_regress_test.csv',
                 'rf_regress_test_2.csv',
                 'svm_regress_test.csv'
                 ]

y_train = data.loadTraining()['ylabels']
x_train = np.zeros((len(trainingDataFiles), len(y_train)))
x_train = np.transpose([loadModelOut(model) for model in trainingDataFiles])

x_test = np.zeros((len(testDataFiles), len(y_train)))
x_test = np.transpose([loadModelOut(model) for model in testDataFiles])


in_layer = len(trainingDataFiles)

parameters = {'hidden_layer_sizes': ((in_layer,), (in_layer, in_layer/2), (in_layer, 2*in_layer/3, in_layer/3)),
              'alpha': np.logspace(-5, -3, 15),
              'learning_rate': ('constant', 'invscaling')
              }

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True, random_state=4)

nn_class = GridSearchCV(estimator=MLPClassifier(), \
    param_grid=dict(parameters), n_jobs=1, cv=None)

nn_class.fit(x_train, y_train)

print 'done fit'

cross_val_scores = cross_validation.cross_val_score(estimator=nn_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=-1)

print "cross val scores: "
print cross_val_scores

y_test = nn_class.predict(x_test)

g = open('nn_blend_params.txt', 'w+')
g.write(str(nn_class.best_estimator_.get_params()))
g.close()


f = open('blended_solution.csv', 'w+')
f.write('Id,Prediction\n')
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(int(y_test[i])) + '\n')
f.close()