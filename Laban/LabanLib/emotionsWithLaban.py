import numpy as np
import LabanUtils.util as labanUtil
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
#from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
import pickle
from sklearn.multiclass import OneVsOneClassifier
from LabanUtils.qualityToEmotion import q2e 
import LabanUtils.combinationsParser as cp

qualities, combinations = cp.getCombinations()
CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen','Michal','Tara']
"""
X,y, featuresNames = labanUtil.accumulateCMA(CMAs, labanUtil.getEmotionsDataset) 
fX = open('X', 'w')
fX.flush()
pickle.dump(X, fX)
fX.close()

ds, featuresNames = labanUtil.accumulatePybrainCMA(CMAs) 
X, Y = labanUtil.getXYfromPybrainDS(ds)
fy = open('Y_Laban', 'w')
fy.flush()
pickle.dump(Y, fy)
fy.close()

"""
X = pickle.load( open( "X", "r" ) )
y = pickle.load( open( "y", "r" ) )
selectedIndices = pickle.load(open( 'selectedIndices', 'r'))
#labanClf = pickle.load(open( 'labanClf', 'r'))
def transform(X, selectedIndices):
    return X[:, selectedIndices]
Y_laban = pickle.load( open( "Y_Laban", "r" ) )
X, y = np.array(X), np.array(y)
baseClf = AdaBoostClassifier()
clf = OneVsOneClassifier(baseClf)
from sklearn import cross_validation
n=1
rs = cross_validation.ShuffleSplit(len(y), n_iter=n, test_size=.1, random_state=0)
res = []
resMixed = []
resLaban = []
for train_index, test_index in rs:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Y_laban_train, Y_laban_test = Y_laban[train_index], Y_laban[test_index] 
    clf.fit(X_train, y_train)#, sample_weight)
    r = clf.score(X_test, y_test)
    res.append(r)
    
    labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X_train, Y_laban_train)
    X_train_transformed = transform(X_train, selectedIndices)
    #X_train_laban = []
    X_train_mixed = []
    for x in X_train_transformed:
        labans = labanClf.predict(x)
        newVec = np.concatenate((x, labans))
        X_train_mixed.append(newVec)
        #X_train_laban.append(labans)
    #X_train_laban=np.array(X_train_laban)
    
    X_test_transformed = transform(X_test, selectedIndices)
    X_test_laban = []
    X_test_mixed = []
    for x in X_test_transformed:
        labans = labanClf.predict(x)
        newVec = np.concatenate((x, labans))
        X_test_mixed.append(newVec)
        X_test_laban.append(labans)
    X_test_laban=np.array(X_test_laban)
    
    clf.fit(X_train_mixed, y_train)
    r = clf.score(X_test_laban, y_test)
    resMixed.append(r)
    
    emotions = ['anger', 'fear', 'happy', 'sad']
    y_test_laban_prediction=[]
    for x in X_test_laban:
        accum = [[],[],[],[]]
        for  q,v in zip(qualities,x):
            es = q2e[q][1]
            emotion = q2e[q][0]
            accum[emotions.index(emotion)].append(es*v)
        grades = []
        for l in accum:
            grades.append(np.mean(l))
        y_test_laban_prediction.append(np.argmax(grades))    
    y_test_laban_prediction = np.array(y_test_laban_prediction)
    diffSize = len(np.nonzero(y_test-y_test_laban_prediction))
    resLaban.append(1-float(diffSize)/len(y_test))
    #print r
print 'clf', str(clf)
print 'n_iter', n
print 'without Laban', np.mean(res)
print res

print 'mixed', np.mean(resMixed)
print resMixed

print 'with Laban', np.mean(resLaban)
print resLaban

    

