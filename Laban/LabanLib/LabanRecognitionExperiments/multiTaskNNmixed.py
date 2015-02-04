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

CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen']
ds, featuresNames = labanUtil.accumulateCMA(CMAs) 
splitProportion=0.1
tstdata, trndata = ds.splitWithProportion( splitProportion )
X, Y = labanUtil.getXYfromPybrainDS(trndata)
X_test, Y_test = labanUtil.getXYfromPybrainDS(tstdata)

selectedFeaureNum = 1000
accum = np.zeros((X.shape[1],))
for y in np.transpose(Y):
    selector = SelectKBest(f_classif, selectedFeaureNum)
    selector = selector.fit(X, y)
    accum += selector.pvalues_
selectedIndices = accum.argsort()[:selectedFeaureNum]
def transform(X):
    return X[:, selectedIndices]     
X, X_test =  transform(X), transform(X_test)
trndata, tstdata = labanUtil.fromXY2DS(X, Y), labanUtil.fromXY2DS(X_test, Y_test)

epochs=2000
hiddenSize=400
initialLearningrate=0.00004
decay=1
myWeightdecay=0
momentum=0.9

net = (labanUtil.constructNet(X.shape[1], hiddenSize, Y.shape[1]))[0]
cvResults = []
trainResults = []
#validResults=[]
testResults = []
totalTrainF1s=[]
totalTestf1 = []
sums=[]
lastTrainVec = np.zeros(Y.shape)
quanta=5
params = range(epochs/quanta)


for epochNum in params:
    validData, trndata = trndata.splitWithProportion( 0.2 )
    X, Y = labanUtil.getXYfromPybrainDS(trndata)
    X_valid, Y_valid = labanUtil.getXYfromPybrainDS(validData)
    trainer = BackpropTrainer(net, dataset=trndata,  
                                learningrate=initialLearningrate,
                                lrdecay=decay, 
                                verbose=True, 
                                weightdecay=myWeightdecay,
                                batchlearning=True,
                                momentum=momentum)
    trainer.trainEpochs(quanta)
    
    trainVec = net.activateOnDataset(trndata)
    validVec = net.activateOnDataset(validData)
    testVec = net.activateOnDataset(tstdata)
    
    trainDif = np.abs(np.subtract(Y, trainVec))
    #validDif = np.abs(np.subtract(Y_valid, validVec))
    testDif = np.abs(np.subtract(Y_test, testVec))
    
    trainRes = float(sum(sum(trainDif)))/Y.shape[0]/Y.shape[1]
    #validRes = float(sum(sum(validDif)))/Y_valid.shape[0]/Y_valid.shape[1]
    testRes = float(sum(sum(testDif)))/Y_test.shape[0]/Y_test.shape[1]
    
    trainResults.append(trainRes)
    #validResults.append(validRes)
    testResults.append(testRes)
    
    splits = []
    for col in range(trainVec.shape[1]):
        bestSplit, bestF1 = labanUtil.getSplitThreshold(trainVec[:, col], Y[:, col])
        splits.append(bestSplit)
    for col in range(trainVec.shape[1]):
        testVec[:, col] = [1 if e>=splits[col] else 0 for e in testVec[:, col]]
        trainVec[:, col] = [1 if e>=splits[col] else 0 for e in trainVec[:, col]]
    trainf1s = []
    testf1s = []
    for j in range(Y.shape[1]):
        trainf1s.append(metrics.f1_score(Y[:,j], trainVec[:,j]))
        testf1s.append(metrics.f1_score(Y_test[:,j], testVec[:,j]))
    totalTrainF1s.append(np.mean(trainf1s))
    totalTestf1.append(np.mean(testf1s))
des='multiTaskNNmixed'\
+', Hidden size: ' + str(hiddenSize)\
+', epochs: '+ str(epochs) \
+', initialLearningrate: '+str(initialLearningrate) \
+', decay: '+str(decay) \
+', momentum: '+str(momentum) \
+'\n myWeightdecay: '+str(myWeightdecay) \
+', best train f1: '+str(max(totalTrainF1s)) \
+', accuracy: '+str(min(trainResults))\
+'\nbest test f1: '+str(max(totalTestf1))\
+', selectedFeaureNum'+ str(selectedFeaureNum)    
plt.plot(params, totalTrainF1s, label='Train F1')
plt.plot(params, totalTestf1, label='Test F1')

plt.plot(params, trainResults, label='Train accuracy')
#plt.plot(params, validResults, label='Valid accuracy')
plt.plot(params, testResults, label='Test accuracy')

plt.legend().draggable()
plt.title(des)
plt.show()    
print des



