from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, SoftmaxLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import BiasUnit
from pybrain.utilities import percentError
import LabanLib.LabanUtils.util as labanUtil
import math
import mocapUtils.interpulation as inter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  
from pybrain.tools.validation import CrossValidator

class Autoencoder:
    
    def fit(self, ds, 
            epochs=100,
            hiddenSize=100, 
            initialLearningrate=0.002,
            decay=0.9999,
            myWeightdecay=0.8,
            plot=False): 
        ds._convertToOneOfMany()
        firstSample = ds.getSample(0)
        inputSize, hiddenSize, outputSize = len(firstSample[0]), hiddenSize, len(firstSample[1])
        inLayer = LinearLayer(inputSize)
        hiddenLayer = LinearLayer(hiddenSize)
        outLayer = SigmoidLayer(outputSize)
        n = FeedForwardNetwork()
        n.addInputModule(inLayer)
        n.addModule(hiddenLayer)
        b = BiasUnit()
        n.addModule(b)
        n.addOutputModule(outLayer)
        in_to_hidden = FullConnection(inLayer, hiddenLayer)
        hidden_to_out = FullConnection(hiddenLayer, outLayer)
        b_to_hidden = FullConnection(b, hiddenLayer)
        b_to_out = FullConnection(b, outLayer)
        n.addConnection(in_to_hidden)
        n.addConnection(hidden_to_out)
        n.addConnection(b_to_hidden)
        n.addConnection(b_to_out)
        n.sortModules()
        print(n.activate([1,2,3]))
        trainer = BackpropTrainer(n, ds,  learningrate=initialLearningrate,\
                            lrdecay=decay, verbose=True, weightdecay=myWeightdecay)
        """
        #trainer.trainEpochs(epochs)
        def eval(net, output, target):
            output = [1 if o>0.5 else 0 for o in output]
            output = np.array(output)
            target = np.array(target)
            assert len(output) == len(target)
            n_correct = sum( output == target )
            return float(n_correct) / float(len(output))
        """
        cv = CrossValidator(trainer, ds,n_folds=2) #valfunc=eval)
        
        Y = np.array([y for x,y in ds])
        if plot:
            prog = []
            for _ in range(epochs):
                trainer.train()
                """
                pred = n.activateOnDataset(ds)
                f1s = []
                for col in range(pred.shape[1]):
                    _, bestF1 = labanUtil.getSplitThreshold(pred[:, col], Y[:, col])
                    f1s.append(bestF1)
                prog.append(np.mean(f1s))
                """
                prog.append(cv.validate())
            plt.plot(range(epochs), prog)
        
        an = FeedForwardNetwork()
        an.addInputModule(inLayer)
        an.addOutputModule(hiddenLayer)
        an.addModule(b)
        an.addConnection(in_to_hidden)
        an.addConnection(b_to_hidden)
        an.sortModules()
        self.net = an

    def transform(self, X):
        transformed = []
        for x in X:
            res = self.net.activate(x)
            transformed.append(res)
        return np.array(transformed)

"""
from pybrain.datasets import ClassificationDataSet
DS = ClassificationDataSet( 3, 2 )
DS.appendLinked( [1,2,3], [0,1] )
DS.appendLinked( [3,2,1], [1,0] )
DS.appendLinked( [1,1,1], [0,0] )
DS.appendLinked( [1,2,1], [1,1] )
DS.appendLinked( [3,5,3], [1,1] )
DS.appendLinked( [2,5,7], [0,1] )
DS.appendLinked( [10,2,0], [1,0] )
an = Autoencoder()
an.fit(DS)


print 'end'
"""

