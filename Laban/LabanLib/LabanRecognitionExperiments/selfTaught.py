import mocapUtils.kinect.angleExtraction as ae
import mocapUtils.kinect.jointsMap as jm

def extractGeneralVec(fileName):
    headers = open(fileName, 'r').readline().split()
    headers = jm.getFileHeader(headers)
    jointsHeaders = headers[2:-4]
    for i in range(len(jointsHeaders)/4):#iterate over joints
        time, xs = ae.getRelative2AncestorPosition(fileName, jointsHeaders[4*i], ver)
