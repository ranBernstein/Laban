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
from sklearn.linear_model import MultiTaskLassoCV, \
    MultiTaskElasticNetCV, MultiTaskElasticNet, MultiTaskLasso

CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen']
ds, featuresNames = labanUtil.accumulateCMA(CMAs) 
splitProportion=0.1
#for e in params:
testNum =10
params = range(500,800,50)
#params = np.linspace(0,1,10)
#params = np.logspace(-11, -6, 20)
precisions = []
recalls = []
testFs = []
trainFs = []
#for selectedFeaureNum in params:
ps = []
rs = []
teFs = []
trFs = []
selectedFeaureNum=500
for test in range(testNum):
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
    #selectedFeaureNum =int(e)
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
    clf = MultiTaskElasticNetCV(normalize=True)
    #clf = MultiTaskLasso(normalize=True)
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
    ps.append(metrics.precision_score(Y_test, pred))
    rs.append(metrics.recall_score(Y_test, pred))
    teF  = metrics.f1_score(Y_test, pred)
    teFs.append(teF)
    trFs.append(metrics.f1_score(Y, predTrain))
    print 'test#: ', test
    p = np.mean(ps)
    r = np.mean(rs)
    f= 2*p*r/(p+r)
    print  p, r, f
#precisions.append(np.mean(ps))
#recalls.append(np.mean(rs))
p = np.mean(ps)
r = np.mean(rs)
f= 2*p*r/(p+r)
print 'selectedFeaureNum: ',selectedFeaureNum
print  p, r, f, np.mean(teFs)
testFs.append(f)
trainFs.append(np.mean(trFs))
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
#des+=' e ' + str(e)
print des
#semilogx
plt.scatter(range(testNum), testFs)
"""
plt.plot(params, trainFs, label='TrainEs F1, max: '+str(max(trainFs)))
plt.plot(params, testFs, label='TestEs F1, max: '+str(max(testFs)))
#plt.plot(params, cvs, label='TestEs F1, max: '+str(max(cvs)))
plt.xlabel('selectedFeaureNum')
plt.legend().draggable()
plt.title(des)
"""
plt.show()
