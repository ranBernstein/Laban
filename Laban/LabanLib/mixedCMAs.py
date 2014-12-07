from sklearn import linear_model
import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics, svm
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from LabanUtils import informationGain as ig 
from sklearn import cross_validation
import copy
import LabanLib.LabanUtils.combinationsParser as cp


def scorer(pipe, X, y):
    pred = pipe.predict(X)
    return metrics.f1_score(y, pred)

def precisionScorer(pipe, X, y):
    pred = pipe.predict(X)
    return metrics.precision_score(y, pred), metrics.recall_score(y, pred)

def recallScorer(pipe, X, y):
    pred = pipe.predict(X)
    return metrics.recall_score(y, pred), metrics.recall_score(y, pred)

if __name__ == '__main__':
    CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen']
    ds, featuresNames = labanUtil.accumulateCMA(CMAs)    
    X, Y = labanUtil.getXYfromPybrainDS(ds)
    c_regulator=20
    qualities, combinations = cp.getCombinations()
    totalScores=[]
    #params = [ 30, 50, 60, 70, 80, 90, 110];
    #params = range(1,5)
    #params = [0.3,0.5,0.6,0.7,0.8,0.9,1]
    filteredFeaturesNum=70
    
    #for filteredFeaturesNum in params:
    clf = svm.LinearSVC(C=c_regulator, loss='LR', penalty='L1', dual=False, 
                        class_weight='auto')#{1: ratio})
    name = str(clf).split()[0].split('(')[0]
    selectedFeaturesNum=0.6*filteredFeaturesNum
    anova_filter = SelectKBest(f_classif, k=filteredFeaturesNum)
    ig_wrapper = SelectKBest(ig.infoGain, k=selectedFeaturesNum)
    fs =[]
    precisions=[]
    recalls =[]
    for i, y in enumerate(np.transpose(Y)):
        pipe = Pipeline([
                        ('filter_selection', anova_filter),
                        ('wrapper_selection', ig_wrapper),
                        ('classification', clf)
                        ])
        pipe.fit(X, y)
        print qualities[i]
        fs+= cross_validation.cross_val_score(pipe, X, y=y,
            scoring=scorer, cv=2, verbose=True,n_jobs=2).tolist()
        precisions+= cross_validation.cross_val_score(pipe, X, y=y,
            scoring=precisionScorer, cv=2, verbose=True,n_jobs=2).tolist()
        recalls+= cross_validation.cross_val_score(pipe, X, y=y,
            scoring=recallScorer, cv=2, verbose=True,n_jobs=2).tolist()
        #fs+=score))
    pre = np.mean(precisions)
    re = np.mean(recalls)
    f = 2*pre*re/(pre+re)
    print 'stage 10'
    print pre, re, f
    plt.show()
    #totalScores.append(np.mean(scores))
    #print totalScores
    #m=np.max(totalScores)
    #print m
    """
    plt.plot(params, totalScores, label=m)
    plt.legend().draggable()
    plt.title('MixedCMAs: '
             +'CLF: '+name \
             + ',CMAs: '+str(CMAs) \
             +'\n C: ' +str(c_regulator))
    plt.xlabel('filteredFeaturesNum')
    plt.ylabel('F1 score')
    plt.show()
    """




