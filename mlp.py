import pkg_resources
import sys
import os 
import sklearn 
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import dataIO as data
import datetime
from grid_search import grid_search
import time

full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

# the 2 values for alpha are needed because there needs to be 2 parameters for
# the grid search.  Could just remove the grid search
parameters = {'hidden_layer_sizes': [tuple([10] * 1)],\
              'activation': ['logistic'],  \
              'learning_rate': ['constant'], \
              'alpha': [7.0] * 2
              }



num_folds = np.prod(np.array([len(parameters[key]) for key in parameters]))
print "Number of folds: " + str(num_folds)

np.random.seed(int(time.clock()*1000000))

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True)

# Our parameter searching function
# x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
#                                 test_size=0.2)

# x2, x3, y2, y3 = cross_validation.train_test_split(x_23, y_23,\
#                                 test_size=0.5)

# nn_class, val_score = grid_search(nnBoostClassifier(), parameters,\
#                               x1, y1, x2, y2)

# print "Validation score: "
# print val_score

# print "Test score: "
# print nn_class.score(x3, y3)


# Grid search CV - make sure cv = # of parameters combos
x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
                                test_size=0.1)

nn_class = GridSearchCV(estimator=MLPClassifier(), \
    param_grid=dict(parameters), n_jobs=2, cv=num_folds)

nn_class.fit(x1, y1)

print "Training score: "
print nn_class.score(x1, y1)

print "Test score : "
print nn_class.score(x_23, y_23)


# cross_val_scores = cross_validation.cross_val_score(estimator=nn_class,\
#     X=x_train, y=y_train, cv=kf_total, n_jobs=1)

# print "cross val scores: "
# print cross_val_scores


x_all_test = data.allTest()['xlabels']

f = open('nn_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = nn_class.predict(x_all_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')
f.close()

g = open('nn_regress_params.txt', 'w+')
g.write(str(nn_class.best_estimator_.get_params()))
g.close()

x_all_train = data.allTrain()['xlabels']

h = open('nn_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_test_est = nn_class.predict(x_all_train)
for i in range(len(y_test_est)):
    h.write(str(i+1) + ',' + str(y_test_est[i]) + '\n')
h.close()