import sklearn 
import numpy as np
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
import dataIO as data

full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

parameters = {'n_estimators': range(5, 51, 30),
              'criterion': ('gini', 'entropy'),
              'max_features': ('auto', 'sqrt', 'log2', 0.2, 0.4, 0.6),
              'max_depth': (3, 5, None),
              'min_samples_split': (2, 5, 8, 11, 14, 17),
              'bootstrap': ('True', 'False')
             }

kf_total = cross_validation.KFold(len(x_train), n_folds=10, shuffle=True,\
    random_state=4)

rf_class = GridSearchCV(estimator=RandomForestClassifier(), \
    param_grid=dict(parameters), n_jobs=1)

rf_class.fit(x_train, y_train)

cross_val_scores = cross_validation.cross_val_score(estimator=rf_class,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('rf_submission.csv', 'w+')
f.write('Id,Prediction\n')

y_test = rf_class.predict(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')

f.close()