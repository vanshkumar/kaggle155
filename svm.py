import sklearn 
import numpy as np
import os 
from sklearn.svm import SVC
from grid_search import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import time
import dataIO as data

full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

parameters = {'C': np.logspace(-2, 2, 10),
              'kernel': ('poly', 'rbf'),
              'degree': range(2, 7),
              # 'epsilon': np.logspace(-2, 0, 10),
              # 'shrinking': (True, False)
              }

np.random.seed(int(time.clock()*1000000))

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True)

x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
                                test_size=0.2)

x2, x3, y2, y3 = cross_validation.train_test_split(x_23, y_23,\
                                test_size=0.5)

svm_class, val_score = grid_search(SVC(), parameters,\
                              x1, y1, x2, y2)

print "Validation score: "
print val_score

print "Test score: "
print svm_class.score(x3, y3)

# svm_class = GridSearchCV(estimator=SVC(), \
#     param_grid=dict(parameters), n_jobs=4, cv=None)

# svm_class.fit(x_train, y_train)

cross_val_scores = cross_validation.cross_val_score(estimator=svm_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('svm_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = svm_class.predict(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')
f.close()

g = open('svm_regress_params.txt', 'w+')
g.write(str(svm_class.get_params()))
g.close()

h = open('svm_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_test_est = svm_class.predict(x_train)
for i in range(len(y_test_est)):
    h.write(str(i+1) + ',' + str(y_test_est[i]) + '\n')
h.close()