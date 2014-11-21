import combinationsParser as cp
from pybrain.datasets import ClassificationDataSet, SupervisedDataSet
import LabanLib.algorithm.generalExtractor as ge
import numpy as np
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure.modules import BiasUnit
import copy
from sklearn import metrics

def getPybrainDataSet(source='Rachelle'):
    first = False#True
    qualities, combinations = cp.getCombinations()
    moods = combinations.keys()
    ds = None
    l=0
    for mood in moods:
        if mood=='neutral':
            continue
        for typeNum in range(1,21):
            for take in range(1,10):
                fileName = 'recordings/'+source+'/'+mood+'/'+\
                str(typeNum)+'_'+str(take)+'.skl'
                try:
                    data, featuresNames = ge.getFeatureVec(fileName, first)
                    print fileName
                    first = False
                except IOError:
                    continue
                if ds is None:#initialization
                    ds = SupervisedDataSet( len(data), len(qualities) )
                output = np.zeros((len(qualities)))
                for q in combinations[mood][typeNum]:
                    output[qualities.index(q)] = 1
                ds.appendLinked(data ,  output)

                l+=sum(output)
    return ds, featuresNames

def constructNet(inLayerSize, hiddenSize, outLayerSize):
    inLayer = LinearLayer(inLayerSize)
    hiddenLayer = SigmoidLayer(hiddenSize)
    outLayer = SigmoidLayer(outLayerSize)
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
    
    return n, inLayer, hiddenLayer, b, in_to_hidden, b_to_hidden

def fromDStoXY(ds):
    outLayerSize = len(ds.getSample(0)[1])
    X=[]
    Y=[]
    for input, tag in ds:
        X.append(input)
        Y.append(tag)
    return np.array(X),np.array(Y)


def getXYforMultiSet(source):
    ds, featuresNames = getPybrainDataSet(source)
    X, Y = fromDStoXY(ds)
    return X, Y, ds

def getXYfromPybrainDS(ds):
    X=[]
    Y=[]
    for x,y in ds:
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y)

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
            bestF1 = f1
    return bestSplit, bestF1

def accumulateCMA(CMAs):
    trndatas = None
    for trainSource in CMAs:
        trndata, featuresNames = getPybrainDataSet(trainSource)  
        print featuresNames[5800]
        if trndatas is None:
            trndatas = trndata
        else:
            for s in trndata:
                print trndatas.indim
                print trndatas.outdim
                print trndata.indim
                print trndata.outdim
                trndatas.appendLinked(*s)
    return trndatas, featuresNames
