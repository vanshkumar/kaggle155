import sklearn 
import numpy as np
import os 
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from grid_search import grid_search
from sklearn import cross_validation
import time
import dataIO as data


full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

parameters = {'C': np.logspace(-8, -5, 15),
              # 'solver' : ['newton-cg', 'lbfgs', 'liblinear']
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

# log_class, val_score = grid_search(LogisticRegression(), parameters,\
#                               x1, y1, x2, y2)

# print "Validation score: "
# print val_score

# print "Test score: "
# print log_class.score(x3, y3)

# Grid search CV - make sure cv = # of parameters combos
x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
                                test_size=0.1)

log_class = GridSearchCV(estimator=LogisticRegression(), \
    param_grid=dict(parameters), n_jobs=2, cv=num_folds)

# log_class = LogisticRegression(C=9e-6)

log_class.fit(x1, y1)

print "Training score: "
print np.mean(y1 == np.array(log_class.predict_proba(x1)[:, 1] >= 0.5, dtype=int))

print "Test score : "
print np.mean(y_23 == np.array(log_class.predict_proba(x_23)[:, 1] >= 0.5, dtype=int))


x_all_test = data.allTest()['xlabels']

f = open('log_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = log_class.predict_proba(x_all_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i][1]) + '\n')
    # f.write(str(i+1) + ',' + str(int(y_test[i][1] >= 0.5)) + '\n')
f.close()

g = open('log_regress_params.txt', 'w+')
g.write(str(log_class.best_estimator_.get_params()))
g.close()

x_all_train = data.allTrain()['xlabels']

h = open('log_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_test_est = log_class.predict_proba(x_all_train)
for i in range(len(y_test_est)):
    h.write(str(i+1) + ',' + str(y_test_est[i][1]) + '\n')
    # h.write(str(i+1) + ',' + str(int(y_test_est[i][1] >= 0.5)) + '\n')
h.close()