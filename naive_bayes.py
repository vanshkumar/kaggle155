import sklearn 
import numpy as np
import os 
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import make_scorer
import dataIO as data


full_train = data.loadTraining()
x_train = full_train['xlabels']
y_train = full_train['ylabels']
full_test  = data.loadTest()
x_test = full_test['xlabels']

kf_total = cross_validation.KFold(len(x_train), n_folds=10,\
      shuffle=True, random_state=4)

gnb = GaussianNB()

gnb.fit(x_train, y_train)

cross_val_scores = cross_validation.cross_val_score(estimator=gnb,\
    X=x_train, y=y_train, cv=kf_total, n_jobs=1)

print "cross val scores: "
print cross_val_scores

f = open('gnb_class.csv', 'w+')
f.write('Id,Prediction\n')

y_test = gnb.predict(x_test)
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(y_test[i]) + '\n')

f.close()