import numpy as np
from mocapUtils.vicon.amcParser import getAMCperiod
import matplotlib.pyplot as plt
import mocapUtils.stitching.stitching.partitionizing as par
import mocapUtils.utils as ut
from cluster import HierarchicalClustering
from mocapUtils.stitching.stitching import MAXIMA_ORDER, CLUSTER_COEFF, plotParts
from scipy.signal import argrelextrema
"""
file = 'AMCs/598.amc'
joint = 'rtibia'
input = getAMCperiod(joint, file)
"""

def getSwingsAndStances(input):
    periods = par.breakToPeriods(input)
    periods = ut.alignByMaxMany(periods)
    stances = []
    swings = []
    for period in periods:
        a = np.array(period)
        localMax = argrelextrema(a, np.greater, 0, MAXIMA_ORDER)
        #localMax is a tuple
        for m in localMax:
            if(len(m) != 1):
                continue
            max = m[0] 
            swings.append(max)
            stances.append(int(len(period)-max))
            break
    return swings, stances
  
