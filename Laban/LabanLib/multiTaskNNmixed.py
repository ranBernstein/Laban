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

CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen']
ds, featuresNames = labanUtil.accumulateCMA(CMAs) 
splitProportion=0.2
tstdata, trndata = ds.splitWithProportion( splitProportion )
X, Y = labanUtil.getXYfromPybrainDS(trndata)
X_test, Y_test = labanUtil.getXYfromPybrainDS(tstdata)

epochs=400
hiddenSize=200
initialLearningrate=0.00012
decay=1
myWeightdecay=100
momentum=0.9

net = (labanUtil.constructNet(X.shape[1], hiddenSize, Y.shape[1]))[0]
trainer = BackpropTrainer(net, dataset=trndata,  
                            learningrate=initialLearningrate,
                            lrdecay=decay, 
                            verbose=True, 
                            weightdecay=myWeightdecay,
                            batchlearning=True,
                            momentum=momentum)
cvResults = []
trainResults = []
totalTrainF1s=[]
totalTestf1 = []
sums=[]
lastTrainVec = np.zeros(Y.shape)
quanta=10
params = range(epochs/quanta)
for epochNum in params:
    trainer.trainEpochs(quanta)
    trainVec = net.activateOnDataset(trndata)
    dif = np.abs(np.subtract(Y, trainVec))
    trainRes = float(sum(sum(dif)))/Y.shape[0]/Y.shape[1]
    trainResults.append(trainRes)
    testVec = net.activateOnDataset(tstdata)
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
+', best test f1: '+str(max(totalTestf1))     
plt.plot(params, totalTrainF1s, label='Train F1')
plt.plot(params, totalTestf1, label='Test F1')
plt.plot(params, trainResults, label='accuracy')
plt.legend().draggable()
plt.title(des)
plt.show()    
print des



