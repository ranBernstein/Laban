import Orange
print Orange.version.version
import LabanUtils.combinationsParser as cp
import numpy as np
import LabanLib.LabanUtils.util as labanUtil

qualities, combinations = cp.getCombinations()
source = 'Rachelle'
ds, featuresNames = labanUtil.getPybrainDataSet(source)

X, Y = labanUtil.fromDStoXY(ds)
Y = [str(e) for y in Y for e in y]
print Y

features = [Orange.feature.Continuous(f) for f in featuresNames]
for q in qualities:
    features.append(Orange.feature.Discrete(q, values=['0','1']))
print features
Domain = Orange.data.Domain(features)
print X.shape, Y.shape
whole = np.concatenate((X,Y), axis=1)
print whole.shape
Table = Orange.data.Table(Domain, whole)
Table.save(source+'.tab')

"""
for x, y in zip(X, Y):
    List, Of, Column, Variables = 
    [Orange.feature.Continuous(x) for x in ['What','Theyre','Called','AsStrings']]



learners = [
    Orange.multilabel.BinaryRelevanceLearner(name="br"),
    Orange.multilabel.BinaryRelevanceLearner(name="br", \
        base_learner=Orange.classification.knn.kNNLearner),
    Orange.multilabel.LabelPowersetLearner(name="lp"),
    Orange.multilabel.LabelPowersetLearner(name="lp", \
        base_learner=Orange.classification.knn.kNNLearner),
    Orange.multilabel.MLkNNLearner(name="mlknn",k=5),
    Orange.multilabel.BRkNNLearner(name="brknn",k=5),
]
data = Orange.data
data = Orange.data.Table("emotions.xml")

res = Orange.evaluation.testing.cross_validation(learners, data,2)
loss = Orange.evaluation.scoring.mlc_hamming_loss(res)
accuracy = Orange.evaluation.scoring.mlc_accuracy(res)
precision = Orange.evaluation.scoring.mlc_precision(res)
recall = Orange.evaluation.scoring.mlc_recall(res)
print 'loss=', loss
print 'accuracy=', accuracy
print 'precision=', precision
print 'recall=', recall
"""