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
    #print source, l, len(ds)
    return ds, featuresNames

def constructNet(inLayerSize, hiddenSize, outLayerSize):
    inLayer = LinearLayer(inLayerSize)
    hiddenLayer = SigmoidLayer(hiddenSize)
    outLayer = LinearLayer(outLayerSize)
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

def fromXY2DS(X, Y):
    ds=SupervisedDataSet(X.shape[1], Y.shape[1])
    for x,y in zip(X,Y):
        ds.addSample(x, y)
    return ds

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

def getSplits(pred, Y):
    splits = []
    for col in range(pred.shape[1]):
        bestSplit, bestF1 = getSplitThreshold(pred[:, col], Y[:, col])
        splits.append(bestSplit)
    return splits

import copy
def quantisizeBySplits(pred_p, splits):
    pred = copy.copy(pred_p)
    for col in range(pred.shape[1]):
        pred[:, col] = [1 if e>=splits[col] else 0 for e in pred[:, col]]
    return np.array(pred)

def accumulateCMA(CMAs):
    trndatas = None
    for trainSource in CMAs:
        trndata, featuresNames = getPybrainDataSet(trainSource)  
        if trndatas is None:
            trndatas = trndata
        else:
            for s in trndata:
                trndatas.appendLinked(*s)
    return trndatas, featuresNames

import os, os.path
def getNonCMAs(nonCMAs, qualities):
    counter = np.zeros((len(qualities)))
    X, Y = [], []
    for nc in nonCMAs:
        dirtocheck = './recordings/'+nc
        for root, _, files in os.walk(dirtocheck):
            for f in files:
                qs = f.split('.')[0]
                qs = qs.split('_')
                y = np.zeros((len(qualities)))
                for q in qs:
                    if q in qualities:
                        y[qualities.index(q)] = 1
                        counter[qualities.index(q)] +=1
                fileName = os.path.join(root, f)
                x, featuresNames = ge.getFeatureVec(fileName, False)
                X.append(x), Y.append(y)
    return np.array(X), np.array(Y), counter

from sklearn.feature_selection import f_classif, SelectKBest, f_regression,RFECV
from sklearn.linear_model import MultiTaskElasticNetCV, MultiTaskElasticNet
def getMultiTaskclassifier(X, Y):
    selectedFeaureNum=500
    accum = np.zeros((X.shape[1],))
    for y in np.transpose(Y):
        selector = SelectKBest(f_classif, selectedFeaureNum)
        selector = selector.fit(X, y)
        accum += selector.pvalues_
    selectedIndices = accum.argsort()[:selectedFeaureNum]
    def transform(X):
        return X[:, selectedIndices]     
    X_filtered = transform(X)
    clf = MultiTaskElasticNetCV(normalize=True)
    clf.fit(X_filtered, Y)
    return clf, transform
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    