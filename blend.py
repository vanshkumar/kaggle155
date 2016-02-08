import sklearn 
import numpy as np
import os 
import dataIO as data
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC



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



parameters = {'C': np.logspace(-4.0, 4.0, 20),
              # 'kernel': ['rbf'],
              'kernel': ['linear', 'poly', 'rbf'],
              # 'degree': [2]
              'degree': np.arange(0.0, 4.0, 1),
              }

kf_total = cross_validation.KFold(len(x_train), n_folds=5,\
      shuffle=True, random_state=4)

svm_class = GridSearchCV(estimator=SVC(), \
    param_grid=dict(parameters), n_jobs=-1, cv=None)

svm_class.fit(x_train, y_train)

print 'done fit'

cross_val_scores = cross_validation.cross_val_score(estimator=svm_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=-1)

print "cross val scores: "
print cross_val_scores

y_test = svm_class.predict(x_test)

g = open('svm_blend_params.txt', 'w+')
g.write(str(svm_class.best_estimator_.get_params()))
g.close()


f = open('blended_solution.csv', 'w+')
f.write('Id,Prediction\n')
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(int(y_test[i])) + '\n')
f.close()