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
data = []
for train_index, test_index in rs:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    Y_laban_train, Y_laban_test = Y_laban[train_index], Y_laban[test_index]   
    labanClf, selectedIndices = labanUtil.getMultiTaskclassifier(X_train, Y_laban_train)
    data.append((labanClf, selectedIndices, train_index, test_index))
f = open('labanClfs', 'w')
f.flush()
pickle.dump(data, f)
f.close()
    

