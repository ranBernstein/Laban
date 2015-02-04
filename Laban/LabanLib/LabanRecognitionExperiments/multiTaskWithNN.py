from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics
import copy
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

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
    return X, np.transpose(Y)
X, Y = getXYforMultiSet(trainSource)
print X
X_test, Y_test = getXYforMultiSet(testSource)
res = []
params = np.linspace(0.001, 0.1, 10)
for p in params:
    print p
    rbm = BernoulliRBM(n_components=int(p*X.shape[1]), n_iter=1000)
    print rbm.fit_transform(X)
    """
    X_small = rbm.fit_transform(X)
    print X_small.shape
    clf = linear_model.MultiTaskElasticNetCV()
    #clf = Pipeline(steps=[('rbm', rbm), ('MultiTaskElasticNetCV', multiClf)])
    clf.fit(X_small, Y)
    print  np.array(clf.predict(rbm.transform(X_test)))
 
    
    predTrain = np.array(clf.predict(X_small))
    splits = []
    for col in range(predTrain.shape[1]):
        splits.append(getSplitThreshold(predTrain[:, col], Y[:, col]))
    pred =  np.array(clf.predict(rbm.transform(X_test)))
    for col in range(pred.shape[1]):
        pred[:, col] = [1 if e>=splits[col] else 0 for e in pred[:, col]]
    r = metrics.f1_score(Y_test, pred)
    res.append(r)
    print r
    print splits
    """

#plt.plot(params, res)
#plt.show()