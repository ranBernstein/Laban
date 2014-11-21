import mocapUtils.kinect.angleExtraction as ae
import mocapUtils.kinect.jointsMap as jm
import numpy as np
import scipy.stats as st
import mocapUtils.interpulation as inter
import mocapUtils.MovingAverage as ma
import LabanLib.analysis.advanceAndRetreate as ar
import LabanLib.analysis.riseAndSink as rs
import LabanLib.analysis.expendingVsCondencing as ec
import LabanLib.analysis.spreadindAndClosing as sp
import LabanLib.analysis.bindVsFree as bf
import LabanLib.analysis.lightVsStrong as ls
import LabanLib.analysis.jump as jump
import LabanLib.LabanUtils.AbstractLabanAnalyzer as aa
import math
chopFactor = 0.0
def getStats(data, label):
    stats = []
    featuresNames = []
    
    stats.append(np.mean(data))
    featuresNames.append(label+' '+', mean')
    
    stats.append(np.mean(np.abs(data)))
    featuresNames.append(label+' '+', mean abs')
    
    stats.append(np.mean(np.square(data)))
    featuresNames.append(label+' '+', mean square')
    
    stats.append(np.std(data))
    featuresNames.append(label+' '+', std')
    
    stats.append(st.skew(data))
    featuresNames.append(label+' '+', skew')
    
    stats.append(st.kurtosis(data))
    featuresNames.append(label+' '+' ,kurtosis')
    
    stats.append(np.max(data))
    featuresNames.append(label+' '+', max')
    
    stats.append(np.min(data))
    featuresNames.append(label+' '+', min')
    
    stats.append(np.max(np.abs(data)))
    featuresNames.append(label+' '+', max abs')
    
    stats.append(np.min(np.abs(data)))
    featuresNames.append(label+' '+', min abs')

    return stats, featuresNames

def analyzeData(time, data, label):    
    v1, f1 = analyzeDataInner(time, data, label)
    """
    firstIndex, lastIndex = pre.chopHeadAndTail(data, chopFactor)
    time, data = time[firstIndex:lastIndex], data[firstIndex:lastIndex]
    v2, f2 = analyzeDataInner(time, data, label+' choped ')
    """
    return v1, f1#v1+v2, f1+f2

def analyzeDataInner(time, data, label):
    vec = []
    featuresNames = []
    v, f = getStats(data, label)
    vec+=v
    featuresNames+=f
    
    _, un = inter.getUniformSampled(time, data)
    cleaned = ma.movingAverage(un, 20, 1.1)
    v, f = getStats(cleaned, label+'after LPF ')
    vec+=v
    featuresNames+=f    
    
    velocity = np.diff(cleaned)
    v, f = getStats(velocity, label+' velocity')
    vec+=v
    featuresNames+=f
    
    acceleration = np.diff(velocity)
    v, f = getStats(acceleration, label+' acceleration')
    vec+=v
    featuresNames+=f
    
    jurk = np.diff(acceleration)
    v, f = getStats(jurk, label+' jurk')
    vec+=v
    featuresNames+=f
    
    return vec, featuresNames

def getFeatureVec(fileName, chopFactor, firstRun=False, joints=None):
    headers = open(fileName, 'r').readline().split()
    #bug in the files
    headers = jm.getFileHeader(headers)
    ver = jm.getVersion(headers)
    featuresNames=[]
    vec=[]
    jointsHeaders = headers[2:-4]
    for i,h in enumerate(jointsHeaders):#drop timestamp, frameNum and floor
        if i%4==3:
            continue
        #time, relJoints = ae.getRelative2AncestorPosition(fileName, h, ver)
        if i%4!=0:
            continue 
        if not joints is None and not h in joints:
            continue
        try:#for joints that don't have father and grandfather 
            time, _, angles, _ = ae.getAngleVec(fileName, h, False, ver)
            if len(angles)==0:
                continue
        except Exception, e:#joint without a father
            continue
        if firstRun:
            print 'getAngleVec', h, len(vec)
        
        v, f = analyzeData(time, angles, h.split('_')[0]+' angle ')
        vec+=v
        featuresNames+=f
    
    for i in range(len(jointsHeaders)/4):#iterate over joints
        
        #Get the joint's relative position
        time, xs = ae.getRelative2AncestorPosition(fileName, jointsHeaders[4*i], ver)
        v, f = analyzeData(time, xs,'Position of '+jointsHeaders[4*i])
        vec+=v
        featuresNames+=f
        
        time, ys = ae.getRelative2AncestorPosition(fileName, jointsHeaders[4*i+1], ver)
        v, f = analyzeData(time, ys,'Position of '+jointsHeaders[4*i+1])
        vec+=v
        featuresNames+=f
        
        time, zs = ae.getRelative2AncestorPosition(fileName, jointsHeaders[4*i+2], ver)
        v, f = analyzeData(time, zs,'Position of '+jointsHeaders[4*i+2])
        vec+=v
        featuresNames+=f
        
        #Get its change
        xs = np.diff(xs)
        ys = np.diff(ys)
        zs = np.diff(zs)
        
        #Make movement vectors
        movements = zip(xs, ys, zs)
        dircetness = []
        if firstRun:
            print 'directness measurement', jointsHeaders[4*i], len(vec)
        dircetnessTime = []
        for j in range(len(movements)-1):
            if ae.length(movements[j]) == 0 or ae.length(movements[j+1]) == 0:
                continue
            first = ae.getUnitVec(movements[j])
            second = ae.getUnitVec(movements[j+1])
            res = np.dot(first, second)
            dircetness.append(res)
            dircetnessTime.append(time[j])
        v, f = analyzeData(dircetnessTime, dircetness, 'Directness ')
        vec+=v
        featuresNames+=f
    extractor = aa.getExtractor(fileName)
    def getStatsAndFeaturesNames(advancements, vec, featuresNames, label, firstRun=False):
        if firstRun:
            print label,  len(vec)
        v, f = getStats(advancements,  label)
        vec+=v
        featuresNames+=f
        
    #AdvanceAndRetreate
    advancements = ar.AdvanceAndRetreate(extractor).extract(fileName)
    label='AdvanceAndRetreate'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)
    
    #RiseAndSink
    advancements = rs.RiseAndSink(extractor).extract(fileName)
    label='RiseAndSink'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)
    
    
    #ExpendingCondencing
    advancements = ec.ExpendingCondencing(extractor).extract(fileName)
    label='Average of Joint Distance From Center'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)

    
    #SpreadindAndClosing
    advancements = sp.SpreadindAndClosing(extractor).extract(fileName)
    label='SpreadindAndClosing'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)
    
    #FreeAndBind
    advancements = bf.FreeAndBind(extractor).extract(fileName)
    label='FreeAndBind'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)

    
    #LightAndStrong
    advancements = ls.LightAndStrong(extractor).extract(fileName)
    label='LightAndStrong'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)

    
    #Jump
    advancements = jump.Jump(extractor).extract(fileName)
    label='Jump'
    getStatsAndFeaturesNames(advancements, vec, featuresNames, label)

    
    #Sudden
    #Is the skew of the velocity of the joints.
    
    return vec, featuresNames










