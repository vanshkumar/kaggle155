import sklearn 
import numpy as np
import os 
import dataIO as data
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import time
from grid_search import grid_search


def loadModelOut(filename, model_size):
    data = np.transpose(np.loadtxt(filename, delimiter=',', skiprows=1, unpack=True))
    return data[model_size * data.shape[0]:, 1]

trainingDataFiles = [
                     'ada_regress_train.csv',
					 # 'knn_regress_train.csv', 
					 'log_regress_train.csv', 
					 'NN_train.csv',
					 'rf_regress_train.csv',
					 # 'svm_regress_train.csv'
					 ]

testDataFiles = [
                 'ada_regress_test.csv',
				 # 'knn_regress_test.csv', 
				 'log_regress_test.csv', 
				 'NN_test.csv',
				 'rf_regress_test.csv',
				 # 'svm_regress_test.csv'
				 ]

model_size = 0.7

y_blend_train = data.loadTrainingBlend()['ylabels']
x_blend_train = np.zeros((len(trainingDataFiles), len(y_blend_train)))
x_blend_train = np.transpose([loadModelOut(model, model_size) for model in trainingDataFiles])

x_all = len(data.allTest()['xlabels'])
x_test = np.zeros((len(testDataFiles), x_all))
x_test = np.transpose([loadModelOut(model, 0) for model in testDataFiles])

# parameters = {'C': np.logspace(-4.0, 4.0, 20),
#               # 'kernel': ['rbf'],
#               'kernel': ['linear', 'poly', 'rbf'],
#               # 'degree': [2]
#               'degree': np.arange(0.0, 4.0, 1),
#               }

parameters = {'C': np.logspace(-2, 2, 10),
              # 'solver' : ['newton-cg', 'lbfgs', 'liblinear']
              }


# parameters = {'n_estimators': (16, 18, 20),
#               # 'criterion': ('gini', 'entropy'),
#               # 'max_features': ('auto', 'sqrt', 'log2'),
#               'min_samples_leaf': [40],
#               'max_depth': (10,),
#               # 'min_samples_split': (50, 68, 85),
#               # 'bootstrap': ('True', 'False')
#              }

num_folds = np.prod(np.array([len(parameters[key]) for key in parameters]))
print "Number of folds: " + str(num_folds)

np.random.seed(int(time.clock()*1000000))

kf_total = cross_validation.KFold(len(x_blend_train), n_folds=10,\
      shuffle=True)

# x_dev, x_val, y_dev, y_val = cross_validation.train_test_split(x_blend_train, y_blend_train,\
#                                 test_size=0.33)

# svm_class, val_score = grid_search(SVC(), parameters, x_dev, y_dev, x_val, y_val)


# x1, x_23, y1, y_23 = cross_validation.train_test_split(x_blend_train, y_blend_train,\
#                                 test_size=0.2, random_state=datetime.time().second)

# x2, x3, y2, y3 = cross_validation.train_test_split(x_23, y_23,\
#                                 test_size=0.5, random_state=datetime.time().second)

# svm_class, val_score = grid_search(SVC(), parameters,\
#                               x1, y1, x2, y2)

# print "Validation score: "
# print val_score

# print "Test score: "
# print svm_class.score(x3, y3)


x1, x_23, y1, y_23 = cross_validation.train_test_split(x_blend_train, y_blend_train,\
                                test_size=0.1)

log_class = GridSearchCV(estimator=LogisticRegression(), \
    param_grid=dict(parameters), n_jobs=1, cv=num_folds)

# rf_class = GridSearchCV(estimator=RandomForestClassifier(), \
#     param_grid=dict(parameters), n_jobs=1, cv=num_folds)

# log_class = LogisticRegression()

log_class.fit(x1, y1)

# rf_class.fit(x1, y1)


# svm_class = SVC(kernel='linear')

# svm_class.fit(x1, y1)

print "Training score: "
print log_class.score(x1, y1)

print "Validation score: "
print log_class.score(x_23, y_23)

# vote_train = np.sum(x1, axis=1) >= 2
# vote_val = np.sum(x_23, axis=1) >= 2

# print "Training score: "
# print np.mean(vote_train == y1)

# print "Validation score: "
# print np.mean(vote_val == y_23)

# svm_class = GridSearchCV(estimator=SVC(), \
#     param_grid=dict(parameters), n_jobs=-1, cv=None)

# svm_class.fit(x_blend_train, y_blend_train)

# print 'done fit'

cross_val_scores = cross_validation.cross_val_score(estimator=log_class,\
    X=x_blend_train, y=y_blend_train, cv=kf_total, n_jobs=-1)

print "min cross val score: "
print np.min(cross_val_scores)

print "mean cross val score: "
print np.mean(cross_val_scores)

print "std dev cross val score: "
print np.std(cross_val_scores)

y_test = log_class.predict(x_test)

g = open('svm_blend_params.txt', 'w+')
g.write(str(log_class.best_estimator_.get_params()))
g.close()

# y_test = np.sum(x_test, axis=1) >= 2

f = open('blended_solution.csv', 'w+')
f.write('Id,Prediction\n')
for i in range(len(y_test)):
    f.write(str(i+1) + ',' + str(int(y_test[i])) + '\n')
f.close()