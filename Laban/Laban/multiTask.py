from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics

CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]
def getXYforMultiSet(source):
    ds, featuresNames = labanUtil.getPybrainDataSet(source)
    X, Y = labanUtil.fromDStoXY(ds)
    return X, np.transpose(Y)
X, Y = getXYforMultiSet(trainSource)
X_test, Y_test = getXYforMultiSet(testSource)

clf = linear_model.MultiTaskElasticNetCV()
clf.fit(X, Y)
pred =  clf.predict(X_test)
print metrics.f1_score(Y_test, pred)
