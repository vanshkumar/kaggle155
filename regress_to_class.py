import dataIO as data
import numpy as np

# File name should not include the extension (but it should be in .csv format in the folder)
# Format of file name should be estimatorname_regressorclass_trainortest
# Examples: ada_regress_train, gnb_class_test
fname = raw_input('Enter file name: ')

regress_data = open(fname+'.csv', 'r+')
class_data   = open(fname[:fname.index('_')] + str('_r2c_') + \
    fname[fname.rindex('_')+1:] + '.csv', 'w+')

regress_data.readline()

y = []

for line in regress_data.readlines():
    vals = line.split(',')
    y.append(float(vals[1][:-1]))

y = np.array(np.array(y) >= 0.5, dtype=int)

class_data.write('Id,Prediction\n')
for i in range(len(y)):
    class_data.write(str(i+1) + ',' + str(y[i]) + '\n')

regress_data.close()
class_data.close()

if 'train' in fname:
    full_train = data.loadTraining()
    x_train = full_train['xlabels']
    y_train = full_train['ylabels']

    print "Thresholded training score is: "
    print np.mean(y_train == y)