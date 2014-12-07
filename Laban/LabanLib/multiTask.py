from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import MultiTaskLasso, MultiTaskElasticNetCV
import algorithm.autoencoder as anModule
import copy
"""
featureVecLen = X_raw.shape[1]
des = an.fit(trndata, hiddenSize=hiddenSize,
       epochs=200, plot=True, 
       initialLearningrate=0.00009, 
       decay=1,#0.999,
       myWeightdecay=10*hiddenSize*featureVecLen,
       testDs=tstdata,
       momentum=0.9)
"""

trainCMAs = ['Rachelle', 'Milca', 'Sharon']
testCMAs = ['Karen']
trndata, featuresNames = labanUtil.accumulateCMA(trainCMAs)    
X_raw, Y = labanUtil.getXYfromPybrainDS(trndata)
tstdata, featuresNames = labanUtil.accumulateCMA(testCMAs)
X_test_raw, Y_test = labanUtil.getXYfromPybrainDS(tstdata)

"""

from pybrain.datasets import SupervisedDataSet
ds = SupervisedDataSet(2, 1)
ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))
trndata =ds
X_raw, Y = labanUtil.getXYfromPybrainDS(trndata)
tstdata = SupervisedDataSet(2, 1)
tstdata.addSample((0, 1), (1,))
tstdata.addSample((1, 0), (1,))
X_test_raw, Y_test = labanUtil.getXYfromPybrainDS(tstdata)
"""

an = anModule.Autoencoder()
hiddenSize=200
epochs=1000
quanta = 50
weightDecay=100
learningrate=0.00015
momentum=0.8
inDim = X_raw.shape[1]
outDim  = Y.shape[1]
an.initSupervised(inDim, outDim, hiddenSize=hiddenSize)
des = 'hiddenSize: '+str(hiddenSize)
des += ', weightDecay: '+str(weightDecay)
des += ', learningrate: '+str(learningrate)
des += ', momentum: '+str(momentum)
des += ', epochs: '+str(epochs)
testFs = []
trainFs = []
netTrainFs = []
lastX = np.zeros((X_raw.shape[0], hiddenSize))

for  i in range(epochs/quanta):
    print 'Epoch: ', i*quanta
    an.trainSupervised(quanta, trndata,
        initialLearningrate=learningrate, 
        decay=1,#0.999,
        myWeightdecay=weightDecay,
        momentum=momentum)
    netTrainFs.append(an.scoreOnDS(trndata))    
    X, X_test = an.transform(X_raw),  an.transform(X_test_raw)
    if (lastX == X).all():
        raise 'problem'
    lastX = copy.deepcopy(X)
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
    
    testFs.append(metrics.f1_score(Y_test, pred))
    trainFs.append(metrics.f1_score(Y, predTrain))
#des+='\n EN test f1: '+ str(testF)
#des+=' , EN train f1: '+ str(trainF)
r = range(epochs/quanta)
plt.plot(r, trainFs, label='TrainEs F1, max: '+str(max(trainFs)))
plt.plot(r, testFs, label='TestEs F1, max: '+str(max(testFs)))
plt.plot(r, netTrainFs, label='train net, max: '+str(max(netTrainFs)))
print des
plt.legend().draggable()
plt.title(des)
plt.show()
