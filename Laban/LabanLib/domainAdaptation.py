import LabanUtils.combinationsParser as cp
import LabanLib.LabanUtils.util as labanUtil
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.multiclass import OneVsRestClassifier
import matplotlib

#X = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
#y = np.array([0,  1])
CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]
c=0.00001
#clf = LogisticRegression(C=c,penalty='l1')
clf = ElasticNetCV()
X, Y, trainDs = labanUtil.getXYforMultiSet(trainSource)
X_test_raw, Y_test, testDs = labanUtil.getXYforMultiSet(testSource)
X_all = np.concatenate((X, X_test_raw))
 
tags = [0]*len(X)+[1]*len(X_test_raw)
clf.fit(X_all, tags)
name = str(clf).split()[0].split('(')[0]

sampleWeights = clf.predict(X)#sample_weight=sampleWeights
CMAs = ['Rachelle', 'Karen']
trainSource = CMAs[0]
testSource = CMAs[1]
withPCA=False
fs=False
c=80
selectedFeaturesNum = 25
trndata, featuresNames = labanUtil.getPybrainDataSet(trainSource)  
tstdata, featuresNames = labanUtil.getPybrainDataSet(testSource)
X, Y = labanUtil.fromDStoXY(trndata)
X_test, Y_test = labanUtil.fromDStoXY(tstdata)
chooser=f_classif
selector = SelectKBest(chooser, selectedFeaturesNum)
clf = svm.LinearSVC(C=c,  loss='LR',# kernel='linear',
                    penalty='L1', dual=False, 
                    class_weight='auto')
f1s=[]
ps =[]
rs=[]
for y, y_test in zip(np.transpose(Y), np.transpose(Y_test)):
    selector.fit(X, y)
    X = selector.transform(X)
    clf.fit(X, y, sample_weight=sampleWeights)
    pred = clf.predict(X_test)
    f1s.append(metrics.f1_score(y, pred))
    ps.append(metrics.precision_score(y, pred))
    rs.append(metrics.recall_score(y, pred))


m = np.mean(f1s)
fig, ax = plt.subplots()
qualities, combinations = cp.getCombinations()
ind = np.arange(len(qualities))
width = 0.25   
f1Rects = ax.bar(ind, f1s, width, color='g', label='F1: '+str(round(np.mean(f1s),3)) )
pRecrs = ax.bar(ind+width, ps, width, color='b', label='Precision: '+str(round(np.mean(ps),3)))
rRects = ax.bar(ind-width, rs, width, color='r', label='Recall: '+str(round(np.mean(rs),3)))
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
         + '\n Features num: '+str(vecLen) \
         + ', Featue selection (FS) method: '+chooser.__name__ \
         + ', Features num after FS: '+str(selectedFeaturesNum) \
         +'\n with C: ' +str(c))
plt.xlabel('Quality')
plt.ylabel('F1 score')
plt.show()