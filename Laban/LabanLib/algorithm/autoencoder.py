from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, GaussianLayer, LSTMLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import BiasUnit
from pybrain.utilities import percentError
import LabanLib.LabanUtils.util as labanUtil
import math
import mocapUtils.interpulation as inter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  
from pybrain.tools.validation import CrossValidator
import copy
class Autoencoder:
    
    def fit(self, ds, 
            epochs=100,
            hiddenSize=100, 
            initialLearningrate=0.002,
            decay=0.9999,
            myWeightdecay=0.8,
            plot=False,
            testDs=None,
            momentum=0): 
        #ds._convertToOneOfMany()
        firstSample = ds.getSample(0)
        print firstSample
        inputSize, hiddenSize, outputSize = len(firstSample[0]), hiddenSize, len(firstSample[1])
        inLayer = LinearLayer(inputSize)
        hiddenLayer =SigmoidLayer(hiddenSize)
        outLayer = LinearLayer(outputSize)
        n = FeedForwardNetwork()
        n.addInputModule(inLayer)
        n.addModule(hiddenLayer)
        b = BiasUnit()
        n.addModule(b)
        n.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        b_to_hidden = FullConnection(b, hiddenLayer)
        b_to_out = FullConnection(b, outLayer)
        n.addConnection(in_to_hidden)
        n.addConnection(hidden_to_out)
        n.addConnection(b_to_hidden)
        n.addConnection(b_to_out)
        n.sortModules()
        trainer = BackpropTrainer(n, ds,  
                            learningrate=initialLearningrate,
                            lrdecay=decay, 
                            verbose=True, 
                            weightdecay=myWeightdecay,
                            batchlearning=True,
                            momentum=momentum)
        """
        #trainer.trainEpochs(epochs)
        def eval(net, output, target):
            output = [1 if o>0.5 else 0 for o in output]
            output = np.array(output)
            target = np.array(target)
            assert len(output) == len(target)
            n_correct = sum( output == target )
            return float(n_correct) / float(len(output))
        """
        cv = CrossValidator(trainer, ds,n_folds=int(len(ds)/10)) #valfunc=eval)
        
        Y = np.array([y for x,y in ds])
        if plot:
            cvResults = []
            trainResults = []
            totalF1s=[]
            totalTestf1 = []
            sums=[]
            lastTrainVec = np.zeros(Y.shape)
            for epochNum in range(epochs):
                trainer.train()
                """
                pred = n.activateOnDataset(ds)
                f1s = []
                for col in range(pred.shape[1]):
                    _, bestF1 = labanUtil.getSplitThreshold(pred[:, col], Y[:, col])
                    f1s.append(bestF1)
                cvResults.append(np.mean(f1s))
                """
                #cvResults.append(cv.validate())
                trainVec = n.activateOnDataset(ds)
                #print trainVec[0]
                dif = np.abs(np.subtract(Y, trainVec))
                difdif = np.abs(np.subtract(lastTrainVec, trainVec))
                lastTrainVec = copy.deepcopy(trainVec)
                #print dif
                trainRes = float(sum(sum(dif)))/Y.shape[0]/Y.shape[1]
                print 'epoch num:', epochNum
                print 'dif sum: ', sum(sum(dif))
                print 'trainVec sum: ', sum(sum(np.abs(trainVec)))
                print 'difdif sum: ', sum(sum(difdif))
                print 'hiddenSize: ', hiddenSize
                print 'initialLearningrate', initialLearningrate
                print 'decay', decay
                print 'myWeightdecay', myWeightdecay
                s = sum(np.abs(trainVec[0]))
                s2 = sum(Y[0])
                print 'sum(trainVec[0])', s
                print 'sum(Y[0])', sum(Y[0])
                trainResults.append(trainRes)
                sums.append(s2)
                splits = []
                for col in range(trainVec.shape[1]):
                    bestSplit, bestF1 = labanUtil.getSplitThreshold(trainVec[:, col], Y[:, col])
                    splits.append(bestSplit)
                if not testDs is None: 
                    testPred = np.array(n.activateOnDataset(testDs))
                    Y_test = np.array([y for x,y in testDs])
                for col in range(trainVec.shape[1]):
                    if not testDs is None: 
                        testPred[:, col] = [1 if e>=splits[col] else 0 for e in testPred[:, col]]
                    trainVec[:, col] = [1 if e>=splits[col] else 0 for e in trainVec[:, col]]
    
                
                f1s = []
                testf1s = []
                for j in range(Y.shape[1]):
                    f1s.append(metrics.f1_score(Y[:,j], trainVec[:,j]))
                    if not testDs is None:
                        testf1s.append(metrics.f1_score(Y_test[:,j], testPred[:,j]))
                totalF1s.append(np.mean(f1s))
                if not testDs is None:
                    totalTestf1.append(np.mean(testf1s))
            #plt.title()
            des= 'Hidden size: ' + str(hiddenSize)\
            +', epochs: '+ str(epochs) \
            +', initialLearningrate: '+str(initialLearningrate) \
            +', decay: '+str(decay) \
            +', momentum: '+str(momentum) \
            +'\n myWeightdecay: '+str(myWeightdecay) \
            +', hidden function: '+hiddenLayer.name \
            +', output function: '+outLayer.name \
            +'\n trainDsSize: '+str(len(ds)) \
            +', testDsSize: '+str(len(testDs)) \
            +', best train f1: '+str(max(totalF1s)) \
            +', accuracy: '+str(min(trainResults))
            if not testDs is None:
                des+= ', best test f1: '+str(max(totalTestf1))
            plt.plot(range(epochs), totalF1s, label='Train F1')
            plt.plot(range(epochs), trainResults, label='accuracy')
            if not testDs is None:
                plt.plot(range(epochs), totalTestf1, label='Test F1')
            plt.legend()
        
        an = FeedForwardNetwork()
        an.addInputModule(inLayer)
        an.addOutputModule(hiddenLayer)
        an.addModule(b)
        an.addConnection(in_to_hidden)
        an.addConnection(b_to_hidden)
        an.sortModules()
        self.net = an
        return des

    def transform(self, X):
        transformed = []
        for x in X:
            res = self.net.activate(x)
            transformed.append(res)
        return np.array(transformed)

"""
from pybrain.datasets import ClassificationDataSet
DS = ClassificationDataSet( 3, 2 )
DS.appendLinked( [1,2,3], [0,1] )
DS.appendLinked( [3,2,1], [1,0] )
DS.appendLinked( [1,1,1], [0,0] )
DS.appendLinked( [1,2,1], [1,1] )
DS.appendLinked( [3,5,3], [1,1] )
DS.appendLinked( [2,5,7], [0,1] )
DS.appendLinked( [10,2,0], [1,0] )
an = Autoencoder()
an.fit(DS)


print 'end'
"""

