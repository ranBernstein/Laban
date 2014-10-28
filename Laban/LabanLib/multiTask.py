from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics
import copy
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNetCV
import algorithm.autoencoder as anModule

def getSplitThreshold(x, y):
    bestSplit = None
    bestF1 = 0
    sortedX= copy.copy(x)
    sortedX.sort()
    splits = []
    for i in range(len(sortedX)-1):
        splits.append((sortedX[i]+sortedX[i+1])/2)
    for split in x:
        newX = [1 if e>=split else 0 for e in x ]
        f1 = metrics.f1_score(newX, y)
        if f1 > bestF1:
            bestSplit = split
    return bestSplit

    
CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]

def getXYforMultiSet(source):
    ds, featuresNames = labanUtil.getPybrainDataSet(source)
    X, Y = labanUtil.fromDStoXY(ds)
    return X, np.transpose(Y), ds

X_raw, Y, trainDs = getXYforMultiSet(trainSource)
X_test_raw, Y_test, testDs = getXYforMultiSet(testSource)

an = anModule.Autoencoder()

params = [int(x) for x in np.linspace(10, 100, 10)]
res = []
for p in params:
    an.fit(trainDs, hiddenSize=p)
    X, X_test = an.transform(X_raw),  an.transform(X_test_raw)
    print p
    clf = MultiTaskElasticNetCV()
    clf.fit(X, Y)
    predTrain = np.array(clf.predict(X))
    splits = []
    for col in range(predTrain.shape[1]):
        splits.append(getSplitThreshold(predTrain[:, col], Y[:, col]))
    pred =  np.array(clf.predict(X_test))
    for col in range(pred.shape[1]):
        pred[:, col] = [1 if e>=splits[col] else 0 for e in pred[:, col]]
    r = metrics.f1_score(Y_test, pred)
    print r
    res.append(r)
"""
selectedFeaturesNum=25
anova_filter = SelectKBest(f_classif, k=selectedFeaturesNum)

plt.spy(coef_multi_task_lasso_)
#pipe = Pipeline([('feature_selection', anova_filter),('classification', clf)])
#pipe.fit(X, Y)
"""
print splits
plt.plot(params, res)
plt.show()
