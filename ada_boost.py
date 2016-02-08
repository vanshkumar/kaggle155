import sklearn 
import numpy as np
import os 
from sklearn.ensemble import AdaBoostRegressor
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

parameters = {'n_estimators': range(5, 50, 5),
              'learning_rate': np.logspace(-3, 1, 15),
              'loss': ('linear', 'square', 'exponential'),
               }

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True, random_state=datetime.time().second)

x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
                                test_size=0.5, random_state=datetime.time().second)

x2, x3, y2, y3 = cross_validation.train_test_split(x_23, y_23,\
                                test_size=0.5, random_state=datetime.time().second)

ada_class, val_score = grid_search(AdaBoostRegressor(), parameters,\
                              x1, y1, x2, y2)

print "Validation score: "
print val_score

print "Test score: "
print ada_class.score(x3, y3)

# ada_class = GridSearchCV(estimator=AdaBoostRegressor(), \
#     param_grid=dict(parameters), n_jobs=1, cv=None)

# ada_class.fit(x_train, y_train)

cross_val_scores = cross_validation.cross_val_score(estimator=ada_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('ada_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = ada_class.predict(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')
f.close()

g = open('ada_regress_params.txt', 'w+')
g.write(str(ada_class.get_params()))
g.close()

h = open('ada_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_test_est = ada_class.predict(x_train)
for i in range(len(y_test_est)):
    h.write(str(i+1) + ',' + str(y_test_est[i]) + '\n')
h.close()