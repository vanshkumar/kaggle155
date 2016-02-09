import sklearn 
import numpy as np
import os 
import dataIO as data
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import time
from grid_search import grid_search


def loadModelOut(filename, model_size):
    data = np.transpose(np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True))
    # print data.shape
    return data[model_size * data.shape[0]:, 1]

trainingDataFiles = [
                     'ada_regress_train.csv',
					 'knn_regress_train.csv', 
					 'log_regress_train.csv', 
					 # 'nn_class_train.csv',
					 'rf_regress_train.csv',
					 'svm_regress_train.csv'
					 ]

testDataFiles = [
                 'ada_regress_test.csv',
				 'knn_regress_test.csv', 
				 'log_regress_test.csv', 
				 # 'nn_class_test.csv',
				 'rf_regress_test.csv',
				 'svm_regress_test.csv'
				 ]

model_size = 0.7

y_blend_train = data.loadTrainingBlend()['ylabels']
x_blend_train = np.zeros((len(trainingDataFiles), len(y_blend_train)))
x_blend_train = np.transpose([loadModelOut(model, model_size) for model in trainingDataFiles])

print x_blend_train.shape
print len(y_blend_train)

x_all = len(data.allTest()['xlabels'])
print x_all
x_test = np.zeros((len(testDataFiles), x_all))
x_test = np.transpose([loadModelOut(model, 0) for model in testDataFiles])

print x_test.shape

parameters = {'C': np.logspace(-4.0, 4.0, 20),
              # 'kernel': ['rbf'],
              'kernel': ['linear', 'poly', 'rbf'],
              # 'degree': [2]
              'degree': np.arange(0.0, 4.0, 1),
              }

np.random.seed(int(time.clock()*1000000))

kf_total = cross_validation.KFold(len(x_blend_train), n_folds=5,\
      shuffle=True)

# x_dev, x_val, y_dev, y_val = cross_validation.train_test_split(x_blend_train, y_blend_train,\
#                                 test_size=0.33)

# svm_class, val_score = grid_search(SVC(), parameters, x_dev, y_dev, x_val, y_val)


# x1, x_23, y1, y_23 = cross_validation.train_test_split(x_blend_train, y_blend_train,\
#                                 test_size=0.2, random_state=datetime.time().second)

# x2, x3, y2, y3 = cross_validation.train_test_split(x_23, y_23,\
#                                 test_size=0.5, random_state=datetime.time().second)

# svm_class, val_score = grid_search(SVC(), parameters,\
#                               x1, y1, x2, y2)

# print "Validation score: "
# print val_score

# print "Test score: "
# print svm_class.score(x3, y3)


x1, x_23, y1, y_23 = cross_validation.train_test_split(x_blend_train, y_blend_train,\
                                test_size=0.15)

svm_class = SVC(kernel='linear', C=0.1)

svm_class.fit(x1, y1)

val_score = svm_class.score(x_23, y_23)

print "Validation score: "
print val_score

# svm_class = GridSearchCV(estimator=SVC(), \
#     param_grid=dict(parameters), n_jobs=-1, cv=None)

# svm_class.fit(x_blend_train, y_blend_train)

print 'done fit'

cross_val_scores = cross_validation.cross_val_score(estimator=svm_class,\
    X=x_blend_train, y=y_blend_train, cv=kf_total, n_jobs=-1)

print "cross val scores: "
print cross_val_scores

y_test = svm_class.predict(x_test)

g = open('svm_blend_params.txt', 'w+')
g.write(str(svm_class.get_params()))
g.close()

f = open('blended_solution.csv', 'w+')
f.write('Id,Prediction\n')
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(int(y_test[i])) + '\n')
f.close()