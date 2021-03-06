from LabanLib.LabanUtils import AbstractLabanAnalyzer
from LabanLib.LabanUtils import AbstractAnalysis
import mocapUtils.kinect.angleExtraction as ae
import mocapUtils.kinect.jointsMap as jm
import numpy as np

class HeadDrop(AbstractAnalysis.AbstractAnalysis):
    
    def getPositiveAndNegetive(self):
        return 'Spreading', 'Closing'
    
    def wrapper(self, lineInFloats, headers, jointsIndices):
        
        return ae.calcAverageDistanceOfIndicesFromLine(lineInFloats, \
                    jointsIndices, *self.extractor.getLongAxeIndices(headers))
        

def analyze(inputFile):
    f = open(file, 'r')
    headers = f.readline()
    headers = jm.getFileHeader(headers)
    drops = []
    for line in f:
        lineInFloats=[float(v) for v in line.split()]
        indexHead = headers.index('Head_Y')
        indexSpine = headers.index('SpineShoulder_Y')
        drops.append(lineInFloats[indexHead] - lineInFloats[indexSpine])
    return np.diff(drops)
