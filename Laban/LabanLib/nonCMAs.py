import LabanLib.LabanUtils.combinationsParser as cp
import numpy as np
import LabanLib.LabanUtils.util as labanUtil

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
clf = labanUtil.getMultiTaskclassifier(X, Y)
splits = labanUtil.getSplits(X, Y)
pred = clf.predict(X_test)
pred = labanUtil.quantisizeBySplits(pred, splits)
from sklearn import metrics
performance = open('nonCMAs.csv', 'w')
performance.flush()
performance.write('Quality, Precision, Recall, F1 score\n')
for q, p, y_test in zip(qualities, np.transpose(pred), np.transpose(Y_test)):
    f = metrics.f1_score(y_test, p)
    p = metrics.precision_score(y_test, p)
    r = metrics.recall_score(y_test, p)
    performance.write(q
                      +', '+ str(round(p,3))\
                      +', '+ str(round(r,3))\
                      +', '+ str(round(f, 3))\
                      +'\n')
performance.close()