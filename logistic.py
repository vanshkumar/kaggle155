import sklearn 
import numpy as np
import os 
from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import dataIO as data

full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

parameters = {'C': [x/10.0 for x in range(2, 30, 15)],
              'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag']
              }

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True, random_state=4)

knn_class = GridSearchCV(estimator=linear_model.LogisticRegression(), \
    param_grid=dict(parameters), n_jobs=1, cv=None)

knn_class.fit(x_train, y_train) 

cross_val_scores = cross_validation.cross_val_score(estimator=knn_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('log_submission.csv', 'w+')
f.write('Id,Prediction\n')

y_test = knn_class.predict_proba(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')

f.close()

g = open('log_params.txt', 'w+')
g.write(str(knn_class.best_estimator_.get_params()))
g.close()
