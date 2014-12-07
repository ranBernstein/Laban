import numpy as np
import LabanUtils.util as labanUtil
import LabanUtils.combinationsParser as cp
import matplotlib.pyplot as plt
from sklearn import metrics, svm
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import Pool
import math
from sklearn.feature_selection import SelectPercentile, f_classif, \
    f_oneway, f_regression, chi2, SelectKBest
from sklearn.pipeline import Pipeline
from LabanUtils import informationGain as ig 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib

#from matplotlib.font_manager import FontProperties
#fontP = FontProperties()
#fontP.set_size('small')
font = {'family' : 'normal',
        'style' : 'italic',
        'size'   : 18}
#legend([plot1], "title", prop = font)
matplotlib.rc('font', **font)

chooser=f_classif
filteredFeaturesNum=40
selectedFeaturesNum=filteredFeaturesNum/4

def eval(ds, clf, splitProportion=0.2, p=4):
    tstdata, trndata = ds.splitWithProportion( splitProportion )
    X, Y = labanUtil.fromDStoXY(trndata)
    X_test, Y_test = labanUtil.fromDStoXY(tstdata)
    f1s={}
    ps ={}
    rs={}
    #for i, (y, y_test) in enumerate(zip(Y, Y_test)):
    anova_filter = SelectKBest(f_classif, k=filteredFeaturesNum)
    ig_wrapper = SelectKBest(ig.infoGain, k=selectedFeaturesNum)
    pipe = Pipeline([
                    #('filter_selection', anova_filter),
                    #('wrapper_selection', ig_wrapper),
                    ('classification', clf)
                    ])
    #pipe = svm.LinearSVC()
    #pipe = KNeighborsClassifier()
    for i, (y, y_test) in enumerate(zip(np.transpose(Y), np.transpose(Y_test))):
        if any(y)==0:
            #f1s.append(1)
            #ps.append(1)
            #rs.append(1)
            continue
        pipe.fit(X, y)
        pred = pipe.predict(X_test)
        name = str(clf).split()[0].split('(')[0]
        #clf.fit(selector.transform(X), y)
        #pred = clf.predict(selector.transform(X_test))
        #f1 = metrics.f1_score(y_test, pred)
        f1s[i] = metrics.f1_score(y_test, pred)
        ps[i] = metrics.precision_score(y_test, pred)
        rs[i] = metrics.recall_score(y_test, pred)

        #f1s.append(f1)
        #ps.append(metrics.precision_score(y_test, pred))
        #rs.append(metrics.recall_score(y_test, pred))
    return f1s, ps, rs

if __name__ == '__main__':
    p = Pool(7)
    qualities, combinations = cp.getCombinations()
    source = 'Sharon' 
    ds, featuresNames = labanUtil.getPybrainDataSet(source)
    inLayerSize = len(ds.getSample(0)[0])
    outLayerSize = len(ds.getSample(0)[1])
    f1s = []
    ps=[] 
    rs=[]
    testNum=200
    ""
    for _ in qualities:
        f1s.append([])
        ps.append([])
        rs.append([])
    m = {}
    #clf = AdaBoostClassifier()
    c_regulator=80
    #clf = svm.LinearSVC()
    clf = svm.LinearSVC(C=c_regulator, loss='LR', dual=False,class_weight='auto')
    #clf = svm.LinearSVC(C=c_regulator, loss='LR', penalty='L1', 
     #                  dual=False, class_weight='auto')#{1: ratio}) 
    splitProportion=0.2
    for i in range(testNum):
        m[i] = p.apply_async(eval, [ds, clf, splitProportion])
    #eval(ds, clf, splitProportion)
    for i in range(testNum):
        cf1s, cps, crs = m[i].get()
        #for i,(f,p,r) in enumerate(zip(cf1s, cps, crs)):
        for i,_ in enumerate(qualities): 
            if i in cf1s:
                f1s[i].append(cf1s[i])
                ps[i].append(cps[i])
                rs[i].append(crs[i])
    realQualities = []
    accumPs = []
    accumRs =[]
    accumFs = []
    for i,q in enumerate(qualities):
        if len(ps[i])==0:
            continue
        realQualities.append(q)
        pm = np.mean(ps[i])
        accumPs.append(pm)
        rm = np.mean(rs[i])
        accumRs.append(rm)
        if pm+rm==0:
            f1=0
        else:
            f1=2*pm*rm/(pm+rm)
        #f1s[i] = np.mean(f1s[i])
        accumFs.append(f1)
        
    m = np.mean(accumFs)
    print m
    print accumFs
    fig, ax = plt.subplots()
    ind = np.arange(len(realQualities))
    width = 0.25   
    print 
    f1Rects = ax.bar(ind, accumFs, width, color='g', label='F1: '+str(np.round(np.mean(accumFs),3)) )
    pRecrs = ax.bar(ind+width, accumPs, width, color='b', label='Precision: '+str(np.round(np.mean(accumPs),3)))
    rRects = ax.bar(ind-width, accumRs, width, color='r', label='Recall: '+str(np.round(np.mean(accumRs),3)))
    ax.set_xticks(ind+width)
    xtickNames = plt.setp(ax, xticklabels=realQualities)
    plt.setp(xtickNames, rotation=90)#, fontsize=8)
    ax.set_xticklabels(realQualities)
    """
    def autolabel(rects):
        # attach some text labels
        for i,rect in enumerate(rects):
            height = rect.get_height()
            #ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, qualities[i],#'%d'%int(height),
             #       ha='center', va='bottom')
    autolabel(f1Rects)
    """
    ax.legend().draggable()
    dsSize = ds.getLength()
    vecLen = len(ds.getSample(0)[0])
    name = str(clf).split()[0].split('(')[0]
    #plt.title('F1 mean: '+str(m)+', Test amount: '+str(testNum))
    plt.title('CLF: '+name
              #+ ', percentile: '+str(percentile) 
              + ', CMA: ' + source
              + ',Dataset size: '+str(dsSize) 
              + ', Input length: ' + str(vecLen)
              +'\nNum of Features after filter: '+str(filteredFeaturesNum)
              + ', Number of features after wrapper: '+str(selectedFeaturesNum)
              + '\nSplit prop: ' +str(splitProportion)
              + ', Repetition number: '+str(testNum)
              + ', stage 7')
    plt.xlabel('Quality')
    #plt.xticks(*enumerate(qualities))
    plt.ylabel('Performance')
    plt.ylim((0,1.2))
    #xtickNames = plt.setp(ax1, xticklabels=qualities)
    #plt.setp(xtickNames, rotation=45, fontsize=8)
    
    plt.show()










