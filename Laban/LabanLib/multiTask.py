from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.pipeline import Pipeline
from LabanUtils import informationGain as ig 
from sklearn import cross_validation
import copy
import LabanLib.LabanUtils.combinationsParser as cp
from sklearn.feature_selection import f_classif, SelectKBest, f_regression,RFECV
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskElasticNetCV

CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen']
ds, featuresNames = labanUtil.accumulateCMA(CMAs) 
splitProportion=0.1
#for e in params:
precisions = []
recalls = []
testFs = []
trainFs = []
testNum =9
params = range(testNum)
for _ in params:
    tstdata, trndata = ds.splitWithProportion( splitProportion )
    #trndata = ds
    X_all, Y_all = labanUtil.getXYfromPybrainDS(ds)
    X, Y = labanUtil.getXYfromPybrainDS(trndata)
    X_test, Y_test = labanUtil.getXYfromPybrainDS(tstdata)
    
    
    def scorer(pipe, X, y):
        pred = pipe.predict(X)
        return metrics.f1_score(y, pred)
    
    #cvs = []
    #params = range(225,265)#np.linspace(0.1, 1, 10)
    #params = np.logspace(-12,-9, 15)
    selectedFeaureNum = 125
    accum = np.zeros((X.shape[1],))
    for y in np.transpose(Y):
        selector = SelectKBest(f_classif, selectedFeaureNum)
        selector = selector.fit(X, y)
        accum += selector.pvalues_
    selectedIndices = accum.argsort()[:selectedFeaureNum]
    def transform(X):
        return X[:, selectedIndices]     
    X_filtered, X_test_filtered, X_all_filtered =  \
    transform(X), transform(X_test), transform(X_all)
    #print e
    clf = MultiTaskElasticNetCV(eps=1e-11)
    #clf = MultiTaskLassoCV(eps=1e-11)
    clf.fit(X_filtered, Y)
    predTrain = np.array(clf.predict(X_filtered))
    splits = []
    for col in range(predTrain.shape[1]):
        bestSplit, bestF1 = labanUtil.getSplitThreshold(predTrain[:, col], Y[:, col])
        splits.append(bestSplit)
    pred =  np.array(clf.predict(X_test_filtered))
    for col in range(pred.shape[1]):
        pred[:, col] = [1 if e>=splits[col] else 0 for e in pred[:, col]]
        predTrain[:, col] = [1 if e>=splits[col] else 0 for e in predTrain[:, col]]
    precisions.append(metrics.precision_score(Y_test, pred))
    recalls.append(metrics.recall_score(Y_test, pred))
    testFs.append(metrics.f1_score(Y_test, pred))
    trainFs.append(metrics.f1_score(Y, predTrain))
    
    #cvs.append(np.mean(cross_validation.cross_val_score(clf, X, y=y,
    #       scoring=scorer, cv=2, verbose=True)))
#print params
print testNum
print precisions
print recalls
p = np.mean(precisions)
r = np.mean(recalls)
f= 2*p*r/(p+r)
print  p, r, f
print testFs
print trainFs
name = str(clf).split()[0].split('(')[0]
des = name+' selectedFeaureNum = '+str(selectedFeaureNum)
des+=' testNum '+str(testNum)
print des
plt.semilogx(params, trainFs, label='TrainEs F1, max: '+str(max(trainFs)))
plt.semilogx(params, testFs, label='TestEs F1, max: '+str(max(testFs)))
#plt.plot(params, cvs, label='TestEs F1, max: '+str(max(cvs)))
plt.xlabel('eps')
plt.legend().draggable()
plt.title(des)
plt.show()
