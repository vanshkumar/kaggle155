import sklearn
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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

parameters = {'n_estimators': range(3, 10, 3),
              # 'criterion': ('gini', 'entropy'),
              # 'max_features': ('auto', 'sqrt', 'log2'),
              'min_samples_leaf': (30, 250),
              'max_depth': (3, 5, 8),
              # 'min_samples_split': (50, 68, 85),
              # 'bootstrap': ('True', 'False')
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

# rf_class, val_score = grid_search(RandomForestClassifier(), parameters,\
#                               x1, y1, x2, y2)

# print "Validation score: "
# print val_score

# print "Test score: "
# print rf_class.score(x3, y3)


# Grid search CV - make sure cv = # of parameters combos
x1, x_23, y1, y_23 = cross_validation.train_test_split(x_train, y_train,\
                                test_size=0.1)

rf_class = GridSearchCV(estimator=RandomForestRegressor(), \
    param_grid=dict(parameters), n_jobs=2, cv=num_folds)

# rf_class = RandomForestClassifier(n_estimators=5, min_samples_leaf=40)

rf_class.fit(x1, y1)

print "Training score: "
print np.mean(y1 == np.array(rf_class.predict(x1) >= 0.5, dtype=int))

print "Test score : "
print np.mean(y_23 == np.array(rf_class.predict(x_23) >= 0.5, dtype=int))


x_all_test = data.allTest()['xlabels']

f = open('rf_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = rf_class.predict(x_all_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')
f.close()

g = open('rf_regress_params.txt', 'w+')
g.write(str(rf_class.best_estimator_.get_params()))
g.close()

x_all_train = data.allTrain()['xlabels']

h = open('rf_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_train_est = rf_class.predict(x_all_train)
for i in range(len(y_train_est)):
    h.write(str(i+1) + ',' + str(y_train_est[i]) + '\n')
h.close()