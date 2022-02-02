import numpy as np

def getMotionAngle(path, robot):
    # PID gains (to tune)
    KpPos = 1
    KpAngle = 2

    previous=abs(path[0][0]-robot[0]) + abs(path[0][1]-robot[1])
    index=0
    for i in range(10,len(path)):
         if abs(path[i][0]-robot[0]) + abs(path[i][1]-robot[1]) <= previous:
            previous=abs(path[i][0]-robot[0]) + abs(path[i][1]-robot[1])
            index=i
    direction=[path[index-10][0]-path[index][0],path[index-10][1]-path[index][1]]
    angleToGoal=np.arctan2(direction[1], direction[0])%(2*np.pi)

    distanceRobotPath = ((path[index][0]-robot[0])**2 + (path[index][1]-robot[1])**2)**(1/2)
    posCorrection=KpPos*distanceRobotPath

    #clip the max posCorrection
    if(posCorrection > 90):
        posCorrection=90

    # check if we're above or under the path
    above=(path[index][0]-robot[0]+np.cos(angleToGoal+0.1))**2+(path[index][1]-robot[1]+np.sin(angleToGoal+0.1))**2
    under=(path[index][0]-robot[0]+np.cos(angleToGoal-0.1))**2+(path[index][1]-robot[1]+np.sin(angleToGoal-0.1))**2
    if(above<under):
        posCorrection=-posCorrection

    #if we're already heading to the path at max angle
    if(abs(angleToGoal-robot[2]) > np.pi/2):
        posCorrection=0

    if((180*(angleToGoal-robot[2]))/np.pi < 180):
        angleCorrection=KpAngle*(180*(angleToGoal-robot[2]))/np.pi
    else:
        angleCorrection=KpAngle*((180*(angleToGoal-robot[2]))/np.pi-360)

    desiredAngleCorrection=angleCorrection+posCorrection
    return desiredAngleCorrection
