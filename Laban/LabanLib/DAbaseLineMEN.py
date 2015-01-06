import LabanLib.LabanUtils.combinationsParser as cp
import numpy as np
import LabanLib.LabanUtils.util as labanUtil
from sklearn import metrics
qualities, combinations = cp.getCombinations()

performance = open('DAbaseLineMEN.csv', 'w')
performance.flush()
performance.write('Quality, TestSubject, Precision, Recall, F1 score\n')

CMAs = ['Rachelle', 'Milca', 'Sharon', 'Karen','Michal','Tara']
for i in range(len(CMAs)):
    trndata,featuresNames = labanUtil.accumulateCMA(CMAs[:i]+CMAs[i+1:])
    X_train, Y_train = labanUtil.getXYfromPybrainDS(trndata)
    clf, transformer = labanUtil.getMultiTaskclassifier(X_train, Y_train)
    X_train_filtered  = transformer(X_train)
    splits = labanUtil.getSplits(clf.predict(X_train_filtered), Y_train)
    
    tstdata,featuresNames = labanUtil.accumulateCMA([CMAs[i]])
    X_test, Y_test = labanUtil.getXYfromPybrainDS(tstdata)
    X_test_filtered = transformer(X_test)
    Pred = clf.predict(X_test_filtered)
    Pred = labanUtil.quantisizeBySplits(Pred, splits)
    for q, pred, y_test in zip(qualities, np.transpose(Pred), np.transpose(Y_test)):
        f = metrics.f1_score(y_test, pred)
        p = metrics.precision_score(y_test, pred)
        r = metrics.recall_score(y_test, pred)
        performance.write(q
                          +', '+CMAs[i]
                          +', '+ str(round(p,3))\
                          +', '+ str(round(r,3))\
                          +', '+ str(round(f, 3))\
                          +'\n')
performance.close()