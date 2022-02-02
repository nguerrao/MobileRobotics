from VisionClass import VisionClass
from GlobalMapClass import GlobalMapClass
import ShorthestPath
from KalmanFilterClass import KalmanFilterClass
from LocalNavigator import LocalNavigator
import motionPlanning
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from tdmclient import ClientAsync, aw


def mapInitialisation():
    print("Map initialization")
    flag=False
    counter=0
    while(not flag and counter < 10):
        flag=True
        vision.update()
        flag&=globalMap.setRobot(vision.robotDetection())
        flag&=globalMap.setGoal(vision.goalDetection())
        globalMap.setObstacles(vision.obstaclesDetection(True))
        counter+=1
        if(not flag):
            time.sleep(1)
            print("failed attempt :",counter)
    if counter >= 10:
        print("Failed map initialization")

    print("Astar running")
    route=ShorthestPath.astar(globalMap.getObstacles(),globalMap.getMapSize()[0], globalMap.getMapSize()[1],
                              globalMap.getRobot(), globalMap.getGoal())
    globalMap.setPath(route)


#for printing purpose only
def displayRoute(route,display=True):
    print("Display route")
    x_coords = []
    y_coords = []
    if(route is not False):
        for i in (range(0,len(route))):
            x = route[i][0]
            y = route[i][1]
            x_coords.append(x)
            y_coords.append(y)

        # plot map and path
    if(display):
        fig, ax = plt.subplots(figsize=(20,20))
        ax = plt.gca()
        ax.invert_yaxis()
        ax.imshow(vision.obstaclesDetection(False), cmap=plt.cm.Dark2)
        ax.imshow(globalMap.getObstacles(), cmap=plt.cm.Dark2, alpha=0.3)
        ax.scatter(globalMap.getRobot()[0],globalMap.getRobot()[1], marker = "*", color = "yellow", s = 200)
        ax.scatter(globalMap.getGoal()[0],globalMap.getGoal()[1], marker = "*", color = "red", s = 200)
        ax.plot(x_coords,y_coords, color = "black")
        plt.show()
    return x_coords, y_coords

print("Variables declaration")
globalMap=GlobalMapClass()
kalmanFilter=KalmanFilterClass()
vision=VisionClass(handCalibration=True)
robot=LocalNavigator()
motorInput=[0,0,0]
goal = False

#Display variables
x_coords = []
y_coords = []
up_width = 1800
up_height = 1200

print("Vision initialize and calibration")
vision.initialize()
time.sleep(5) #to get the of the camera done

# create a new map, run the path planning and initialize kalman filter with the robot position
mapInitialisation()
kalmanFilter.setState(globalMap.getRobot())
x_coords, y_coords=displayRoute(globalMap.getPath())

#  Create and setup a window, display purpose only
cv2.startWindowThread()
cv2.imshow('Robot', vision.imageDraw)
cv2.resizeWindow('Robot', up_width, up_height)
up_points = (up_width, up_height)
resized_up = cv2.resize(vision.imageDraw, up_points, interpolation= cv2.INTER_LINEAR)
cv2.imshow('Robot', vision.imageDraw)
cv2.resizeWindow('Robot', up_width, up_height)

# navigation algorithm start
print("Start navigation")
while(not goal):
    # get robot position with kalmanFilter
    robotPos=kalmanFilter.predict(motorInput,0.1)
    vision.update()
    meas = vision.robotDetection()
    if meas is not False:
        robotPos=kalmanFilter.update(meas)
    globalMap.setRobot(robotPos)

    #Check if it has reached the goal
    if  abs(globalMap.getGoal()[0]-globalMap.getRobot()[0])+abs(globalMap.getGoal()[1]-globalMap.getRobot()[1]) < 30:
        goal = True

    # Make the robot move
    motorSpeed, omega, kidnap = aw(robot.run(motionPlanning.getMotionAngle(globalMap.getPath(),globalMap.getRobot())))
    motorInput=[motorSpeed*np.cos(globalMap.getRobot()[2])/3.2,motorSpeed*np.sin(globalMap.getRobot()[2])/3.2, -1.1*(omega*np.pi/180)]

    # Check if the robot was kidnapped
    if(kidnap):
        print("Kidnap")
        aw(robot.stop())
        time.sleep(10)
        mapInitialisation()
        x_coords, y_coords=displayRoute(globalMap.getPath(),False)

    # Displaying
    image = vision.imageDraw
    cv2.circle(image, (int(globalMap.getGoal()[0]), int(globalMap.getGoal()[1])), 15, (0, 0, 255), 1)
    cv2.circle(image, (int(globalMap.getRobot()[0]), int(globalMap.getRobot()[1])), 15, (255, 0, 0), 1)
    cv2.arrowedLine(image,(int(globalMap.getRobot()[0]), int(globalMap.getRobot()[1])),(int(globalMap.getRobot()[0]+30*np.cos(globalMap.getRobot()[2])),
                    int(globalMap.getRobot()[1]+30*np.sin(globalMap.getRobot()[2]))),color=(255, 0, 0),thickness=1, tipLength=0.2)
    for i in range(len(x_coords)):
        cv2.circle(image, (x_coords[i], y_coords[i]), 1, (0, 0, 255), 2)
    resized_up = cv2.resize(vision.image, up_points, interpolation= cv2.INTER_LINEAR)
    cv2.putText(resized_up, "angle{:d}, robot : x {:d}, y {:d}".format(int(motionPlanning.getMotionAngle(globalMap.getPath(),globalMap.getRobot())),int(globalMap.getRobot()[0]),int(globalMap.getRobot()[1])),(5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('Robot', resized_up)
    cv2.waitKey(1)

    time.sleep(0.1)

# Goal is reached, end
aw(robot.stop())
print("Goal reached")
time.sleep(15)
vision.finish()
