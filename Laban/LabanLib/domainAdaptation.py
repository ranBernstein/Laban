import LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil
import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.multiclass import OneVsRestClassifier
import matplotlib
trainCMAs = ['Milca', 'Rachelle' , 'Sharon']
trndata, featuresNames = labanUtil.accumulateCMA(trainCMAs) 

testCMAs = ['Karen']
tstdata, featuresNames = labanUtil.accumulateCMA(testCMAs) 
  
selectedFeaturesNum = 150
X, Y = labanUtil.fromDStoXY(trndata)
X_test, Y_test = labanUtil.fromDStoXY(tstdata)
X_all = np.concatenate((X, X_test))
tags = [0]*len(X)+[1]*len(X_test)
adaptor = LogisticRegression()
adaptorName = str(adaptor).split()[0].split('(')[0]
filter=f_classif
selector = SelectKBest(filter, selectedFeaturesNum)
"""
clf = svm.LinearSVC(C=c,  loss='LR',# kernel='linear',
                    penalty='L1', dual=False, 
                    class_weight='auto')
"""
lr=1
ne=38
clf = AdaBoostClassifier(learning_rate=lr, n_estimators=ne)
clf_weights = AdaBoostClassifier(learning_rate=lr, n_estimators=ne)
#c = svm.SVC(pena)
#clf_weights = svm.SVC()
#clf = svm.SVC()
f1s=[]
ps =[]
rs=[]
weightF1s=[]
trainWeightF1s = []
for y, y_test in zip(np.transpose(Y), np.transpose(Y_test)):
    selector.fit(X, y)
    
    X_all_filtered = selector.transform(X_all)
    adaptor.fit(X_all_filtered, tags)
    X_filtered = selector.transform(X)
    X_test_filtered = selector.transform(X_test)
    """
    lowestSampleValue = 0
    sampleWeights = np.array(adaptor.predict(X_filtered))
    sampleWeights -= np.min(sampleWeights)
    sampleWeights += lowestSampleValue
    """    
    sampleWeights = adaptor.predict_proba(X_filtered)[:,1]
    clf_weights.fit(X_filtered, y, sample_weight=sampleWeights)
    weightPred = clf_weights.predict(X_test_filtered)
    weightF1s.append(metrics.f1_score(y_test, weightPred))
    
    trainWeightPred = clf_weights.predict(X_filtered)    
    trainWeightF1s.append(metrics.f1_score(y, trainWeightPred))
    
    #ps.append(metrics.precision_score(y, pred))
    #rs.append(metrics.recall_score(y, pred))
    
    clf.fit(X_filtered, y)
    pred = clf.predict(X_test_filtered)
    f1s.append(metrics.f1_score(y_test, pred))
    #ps.append(metrics.precision_score(y, pred))
    #rs.append(metrics.recall_score(y, pred))


m = np.mean(f1s)
fig, ax = plt.subplots()
qualities, combinations = cp.getCombinations()
ind = np.arange(len(qualities))
width = 0.25   
f1Rects = ax.bar(ind, f1s, width, color='g', label='F1: '+str(round(np.mean(f1s),3)) )
weightF1Rects = ax.bar(ind+width, weightF1s, width, color='b', label='WeightF1: '+str(round(np.mean(weightF1s),3)) )
ax.bar(ind+2*width, trainWeightF1s, width, color='r', \
       label='trainWeight: '+str(round(np.mean(trainWeightF1s),3)) )
#pRecrs = ax.bar(ind+width, ps, width, color='b', label='Precision: '+str(round(np.mean(ps),3)))
#rRects = ax.bar(ind-width, rs, width, color='r', label='Recall: '+str(round(np.mean(rs),3)))
ax.set_xticks(ind)
#ax.set_xticklabels(ind)
xtickNames = plt.setp(ax, xticklabels=qualities)
plt.setp(xtickNames, rotation=90)#, fontsize=8)
ax.set_xticklabels(qualities)

ax.legend().draggable()
trainSize = trndata.getLength()
testSize = tstdata.getLength()
vecLen = len(trndata.getSample(0)[0])
name = str(clf).split()[0].split('(')[0]
adaptorName = str(adaptor).split()[0].split('(')[0]
#plt.title('F1 mean: '+str(m)+', Test amount: '+str(testNum))
font = {'family' : 'normal',
        'style' : 'italic',
        'size'   : 20}
#legend([plot1], "title", prop = font)
matplotlib.rc('font', **font)

plt.title('CLF: '+name \
         + 'learning_rate: '+str(lr) \
         + 'n_estimators: '+str(ne) \
         + '\n adaptor '+ adaptorName\
         + ', Features num after FS: '+str(selectedFeaturesNum))
plt.xlabel('Quality')
plt.ylabel('F1 score')
plt.show()