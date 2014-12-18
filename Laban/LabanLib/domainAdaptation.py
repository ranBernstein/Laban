import LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil
import numpy as np
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.multiclass import OneVsRestClassifier
import matplotlib

CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]
c=0.00001
 

withPCA=False
fs=False
c=80
selectedFeaturesNum = 70
trndata, featuresNames = labanUtil.getPybrainDataSet(trainSource)  
tstdata, featuresNames = labanUtil.getPybrainDataSet(testSource)
X, Y = labanUtil.fromDStoXY(trndata)
X_test, Y_test = labanUtil.fromDStoXY(tstdata)
X_all = np.concatenate((X, X_test))
tags = [0]*len(X)+[1]*len(X_test)
adaptor = ElasticNetCV()
adaptorName = str(adaptor).split()[0].split('(')[0]
filter=f_classif
selector = SelectKBest(filter, selectedFeaturesNum)
"""
clf = svm.LinearSVC(C=c,  loss='LR',# kernel='linear',
                    penalty='L1', dual=False, 
                    class_weight='auto')
"""
clf = AdaBoostClassifier()
clf_weights = AdaBoostClassifier()
#c = svm.SVC(pena)
#clf_weights = svm.SVC()
#clf = svm.SVC()
f1s=[]
ps =[]
rs=[]
weightF1s=[]
for y, y_test in zip(np.transpose(Y), np.transpose(Y_test)):
    selector.fit(X, y)
    
    X_all_filtered = selector.transform(X_all)
    adaptor.fit(X_all_filtered, tags)
    X_filtered = selector.transform(X)
    X_test_filtered = selector.transform(X_test)
    #print 'adaptor.predict(X_filtered)'
    #print adaptor.predict(X_filtered)
    #print 'adaptor.predict(X_test_filtered)'
    #print adaptor.predict(X_test_filtered)
    lowestSampleValue = 0.5
    sampleWeights = np.array(adaptor.predict(X_filtered))
    sampleWeights -= np.min(sampleWeights)
    sampleWeights += lowestSampleValue
    clf_weights.fit(X_filtered, y, sample_weight=sampleWeights)
    weightPred = clf_weights.predict(X_test_filtered)
    weightF1s.append(metrics.f1_score(y_test, weightPred))
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
width = 0.3   
f1Rects = ax.bar(ind, f1s, width, color='g', label='F1: '+str(round(np.mean(f1s),3)) )
weightF1Rects = ax.bar(ind+width, weightF1s, width, color='b', label='WeightF1: '+str(round(np.mean(weightF1s),3)) )
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
#plt.title('F1 mean: '+str(m)+', Test amount: '+str(testNum))
font = {'family' : 'normal',
        'style' : 'italic',
        'size'   : 20}
#legend([plot1], "title", prop = font)
matplotlib.rc('font', **font)

plt.title('CLF: '+name \
         + ', Train set: CMA #1'+' size-'+str(trainSize) \
         + ', Test set: CMA #2'+' size-'+str(testSize) \
         + '\n lowestSampleValue '+str(lowestSampleValue) \
         + ', Featue selection (FS) method: '+filter.__name__ \
         + ', Features num after FS: '+str(selectedFeaturesNum))
plt.xlabel('Quality')
plt.ylabel('F1 score')
plt.show()