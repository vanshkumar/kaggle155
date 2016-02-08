import sklearn 
import numpy as np
import os 
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from grid_search import grid_search
from sklearn import cross_validation
import datetime
import dataIO as data

full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

parameters = {'C': np.logspace(-4, 1.5, 20),
              'solver' : ['newton-cg', 'lbfgs', 'liblinear']
              }

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True, random_state=datetime.time().second)

x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
                                test_size=0.5, random_state=datetime.time().second)

x2, x3, y2, y3 = cross_validation.train_test_split(x_23, y_23,\
                                test_size=0.5, random_state=datetime.time().second)

log_class, val_score = grid_search(LogisticRegression(), parameters,\
                              x1, y1, x2, y2)

print "Validation score: "
print val_score

print "Test score: "
print log_class.score(x3, y3)

# log_class = GridSearchCV(estimator=LogisticRegression(), \
#     param_grid=dict(parameters), n_jobs=1, cv=None)

# log_class.fit(x_train, y_train)

cross_val_scores = cross_validation.cross_val_score(estimator=log_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('log_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = log_class.predict_proba(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i][1]) + '\n')
f.close()

g = open('log_regress_params.txt', 'w+')
g.write(str(log_class.get_params()))
g.close()

h = open('log_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_test_est = log_class.predict_proba(x_train)
for i in range(len(y_test_est)):
    h.write(str(i+1) + ',' + str(y_test_est[i][1]) + '\n')
h.close()