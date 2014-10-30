import LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt

#X = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
#y = np.array([0,  1])
CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]
c=0.00001
#clf = LogisticRegression(C=c,penalty='l1')
clf = ElasticNetCV()
X_raw, Y, trainDs = labanUtil.getXYforMultiSet(trainSource)
X_test_raw, Y_test, testDs = labanUtil.getXYforMultiSet(testSource)
X_all = np.concatenate((X_raw, X_test_raw))
 
tags = [0]*len(X_raw)+[1]*len(X_test_raw)
clf.fit(X_all, tags)
name = str(clf).split()[0].split('(')[0]
print name
print c
print clf.predict(X_all)

plt.scatter(range(len(X_raw)), clf.predict(X_raw), c='b', label =trainSource)
plt.scatter(range(len(X_raw), len(X_raw)+len(X_test_raw)), clf.predict(X_test_raw), c='r', label=testSource)
plt.xlabel('Sample #')
plt.ylabel('Regression result')
plt.legend().draggable()
plt.title('Separation between two CMA unlabeled samples')
plt.show()




def getWeightsVector(X_raw, X_test_raw):
    X_all = np.concatenate((X_raw, X_test_raw[:-1]))
    clf= ElasticNetCV()
    clf.fit(X_all, np.zeros((len(X_raw), 1))+np.ones((len(X_test_raw[:-1]), 1)))