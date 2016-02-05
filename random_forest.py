import sklearn 
import numpy as np
import os 
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import dataIO as data

full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

parameters = {'n_estimators': range(5, 50, 1),
              # 'criterion': ('gini', 'entropy'),
              'max_features': ('auto', 'sqrt', 'log2'),
              'max_depth': (3, 5, None),
              'min_samples_split': (2, 5, 8, 11, 14, 17),
              'bootstrap': ('True', 'False')
             }

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True, random_state=4)

rf_class = GridSearchCV(estimator=RandomForestRegressor(), \
    param_grid=dict(parameters), n_jobs=1, cv=None)

rf_class.fit(x_train, y_train)

cross_val_scores = cross_validation.cross_val_score(estimator=rf_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('rf_regress_test.csv', 'w+')
f.write('Id,Prediction\n')
y_test = rf_class.predict(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')
f.close()

g = open('rf_regress_params.txt', 'w+')
g.write(str(rf_class.best_estimator_.get_params()))
g.close()

h = open('rf_regress_train.csv', 'w+')
h.write('Id,Prediction\n')
y_train_est = rf_class.predict(x_train)
for i in range(len(y_train_est)):
  h.write(str(i+1) + ',' + str(y_train_est[i]) + '\n')
h.close()