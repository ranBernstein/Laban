import numpy as np
import LabanLib.LabanUtils.util as labanUtil
import LabanLib.LabanUtils.combinationsParser as cp
import LabanLib.algorithm.generalExtractor as ge
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool
import math
from sklearn.feature_selection import f_classif, SelectKBest, f_regression,RFECV
from sklearn.pipeline import Pipeline
import LabanLib.LabanUtils.informationGain as ig
import matplotlib

chooser=f_classif#ig.infoGain#ig.recursiveRanking#
trainCMAs = ['Milca', 'Rachelle' , 'Sharon']
trndata, featuresNames = labanUtil.accumulateCMA(trainCMAs) 

testCMAs = ['Karen']
tstdata, featuresNames = labanUtil.accumulateCMA(testCMAs) 
  
fs=False
c_regulator=20
ratio ='auto'
#percentile=5
clf = svm.LinearSVC(C=c_regulator,  loss='LR', penalty='L1',\
                     dual=False, class_weight='auto')#{1: ratio})

X, Y = labanUtil.fromDStoXY(trndata)
X_test, Y_test = labanUtil.fromDStoXY(tstdata)
f1s=[]
ps =[]
rs=[]
filteredFeaturesNum = 70
selectedFeaturesNum=0.6*filteredFeaturesNum
qualities, combinations = cp.getCombinations()
anova_filter = SelectKBest(chooser, k=selectedFeaturesNum)
ig_wrapper = SelectKBest(ig.infoGain, k=selectedFeaturesNum)
for i, (y, y_test) in enumerate(zip(np.transpose(Y), np.transpose(Y_test))):
    if all(v == 0 for v in y):
        continue
    pipe = Pipeline([
                    ('feature_selection', anova_filter),
                    ('wrapper_selection', ig_wrapper),
                    ('classification', clf)
                    ])
    pipe.fit(X, y)
    
    predTrain =  pipe.predict(X)
    f1sTrain = metrics.f1_score(y, predTrain)
    pred = pipe.predict(X_test)
    
    precision = metrics.precision_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    
    selector = SelectKBest(chooser, 1)
    selector.fit(X, y)
    featureNum = selector.get_support().tolist().index(True)
    pstr = str(selector.pvalues_[featureNum])
    pstr = pstr[:3] + pstr[-4:]
    scoreStr = str(round(selector.scores_[featureNum],2))
    f1s.append(f1)
    ps.append(precision)
    rs.append(recall)
    print qualities[i], precision, recall, f1
    name = str(clf).split()[0].split('(')[0]


m = np.mean(f1s)
print m, ge.chopFactor
fig, ax = plt.subplots()
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
         + ', Train set: '+str(trainCMAs) \
         + ', Test set'+str(testCMAs) \
         + '\n Features num: '+str(vecLen) \
         + ', Featue selection (FS) method: '+chooser.__name__ \
         + ', Features num after FS: '+str(selectedFeaturesNum) \
         #+', chopFactor: '+str(ge.chopFactor)
         #+', with PCA: ' +str(withPCA)
         #+', with fs: ' +str(fs)
         +'\n with C: ' +str(c_regulator)
         +', cw: '+str(ratio))
plt.xlabel('Quality')
plt.ylabel('F1 score')
plt.show()
