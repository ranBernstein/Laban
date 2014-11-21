from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNetCV
import algorithm.autoencoder as anModule

"""   
CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]

X_raw, Y, trainDs = labanUtil.getXYforMultiSet(trainSource)
X_test_raw, Y_test, testDs = labanUtil.getXYforMultiSet(testSource)
"""
trainCMAs = ['Rachelle', 'Milca']
testCMAs = ['Karen']
trndata, featuresNames = labanUtil.accumulateCMA(trainCMAs)    
X_raw, Y = labanUtil.getXYfromPybrainDS(trndata)
tstdata, featuresNames = labanUtil.accumulateCMA(testCMAs)
X_test_raw, Y_test = labanUtil.getXYfromPybrainDS(tstdata)
an = anModule.Autoencoder()

params = [int(x) for x in np.linspace(10, 100, 10)]
res = []
#for p in params:
#print p
des = an.fit(trndata, hiddenSize=200,
       epochs=50, plot=True, 
       initialLearningrate=0.0001, 
       decay=1,#0.999,
       myWeightdecay=0,
       testDs=tstdata,
       momentum=0)

X, X_test = an.transform(X_raw),  an.transform(X_test_raw)
clf = MultiTaskElasticNetCV()
clf.fit(X, Y)
predTrain = np.array(clf.predict(X))
splits = []
for col in range(predTrain.shape[1]):
    bestSplit, bestF1 = labanUtil.getSplitThreshold(predTrain[:, col], Y[:, col])
    splits.append(bestSplit)
pred =  np.array(clf.predict(X_test))
for col in range(pred.shape[1]):
    pred[:, col] = [1 if e>=splits[col] else 0 for e in pred[:, col]]
    predTrain[:, col] = [1 if e>=splits[col] else 0 for e in predTrain[:, col]]
    
"""
selectedFeaturesNum=25
anova_filter = SelectKBest(f_classif, k=selectedFeaturesNum)

plt.spy(coef_multi_task_lasso_)
#pipe = Pipeline([('feature_selection', anova_filter),('classification', clf)])
#pipe.fit(X, Y)
"""

#plt.plot(params, res)
testF = metrics.f1_score(Y_test, pred)
trainF = metrics.f1_score(Y, predTrain)

des+='\n EN test f1: '+ str(testF)
des+=' , EN train f1: '+ str(trainF)
print des
plt.title(des)
plt.show()
