from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import BiasUnit
from pybrain.utilities import percentError
import LabanLib.LabanUtils.util as labanUtil
import math
import utils.interpulation as inter
import numpy as np
import matplotlib.pyplot as plt


class Autoencoder:
    
    def fit(self, ds, 
            epochs=100,
            hiddenSize=100, 
            initialLearningrate=0.002,
            decay=0.9999,
            myWeightdecay=0.8): 
        ds._convertToOneOfMany()
        firstSample = ds.getSample(0)
        inputSize, hiddenSize, outputSize = len(firstSample[0]), hiddenSize, len(firstSample[1])
        inLayer = LinearLayer(inputSize)
        hiddenLayer = SigmoidLayer(hiddenSize)
        outLayer = LinearLayer(outputSize)
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
        trainer = BackpropTrainer(n, ds,  learningrate=initialLearningrate,\
                            lrdecay=decay, verbose=True, weightdecay=myWeightdecay)

        trainer.trainEpochs(epochs)
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








