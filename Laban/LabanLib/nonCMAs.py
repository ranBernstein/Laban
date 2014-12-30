import LabanLib.LabanUtils.combinationsParser as cp
import numpy as np
import LabanLib.LabanUtils.util as labanUtil
print 'kuku'
qualities, combinations = cp.getCombinations()
nonCMAs = ['Gal','Ayelet']

X_test, Y_test, counter = labanUtil.getNonCMAs(nonCMAs, qualities)
print zip(qualities, counter)

"""
f = open('nonCMAs', 'w')
f.flush()
for x,y in zip(X, Y):
    f.write()
"""
CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen','Michal','Tara']
ds, featuresNames = labanUtil.accumulateCMA(CMAs) 
X, Y = labanUtil.getXYfromPybrainDS(ds)
clf, transformer = labanUtil.getMultiTaskclassifier(X, Y)
X_filtered  = transformer(X)
splits = labanUtil.getSplits(clf.predict(X_filtered), Y)
X_test_filtered = transformer(X_test)
Pred = clf.predict(X_test_filtered)
Pred = labanUtil.quantisizeBySplits(Pred, splits)
from sklearn import metrics
performance = open('nonCMAs.csv', 'w')
performance.flush()
performance.write('Quality, Precision, Recall, F1 score\n')
for q, pred, y_test in zip(qualities, np.transpose(Pred), np.transpose(Y_test)):
    f = metrics.f1_score(y_test, pred)
    p = metrics.precision_score(y_test, pred)
    r = metrics.recall_score(y_test, pred)
    performance.write(q
                      +', '+ str(round(p,3))\
                      +', '+ str(round(r,3))\
                      +', '+ str(round(f, 3))\
                      +'\n')
performance.close()