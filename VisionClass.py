import cv2
import numpy as np
import calibrate

class VisionClass(object):
    def __init__(self, handCalibration):
        self.image=None
        self.imageDraw=None
        self.VideoCap=None
        self.lowGoal=np.array([37, 14, 90])
        self.highGoal=np.array([97, 74, 150])
        self.lowRobot=np.array([75, 60, 60])
        self.highRobot=np.array([95,255,255])
        self.HSVG=False
        self.HSVR=True
        self.handCalibration=handCalibration
        self.tresh=50


    def initialize(self):
        """
        This function defines the bounds for the filters that will be used to detect 
        the colour of the robot and the goal in the case of a manual calibration which
        determined the values of the image directly (then we just need to do a range of
        values)

        """
        if(self.handCalibration):
            G1, G2, G3, HSVG, R1, R2, R3, HSVR, tresh=calibrate.calibration()
            self.tresh=tresh
            self.HSVG=HSVG
            self.HSVR=HSVR
            if(HSVG):
                self.lowGoal=np.array([G1-10,G2,G3])
                self.highGoal=np.array([G1+10,255,255])
            else:
                self.lowGoal=np.array([G1-30,G2-30,G3-30])
                self.highGoal=np.array([G1+30,G2+30,G3+30])
            if(HSVR):
                self.lowRobot=np.array([R1-15,R2,R3])
                self.highRobot=np.array([R1+15,255,255])
            else:
                self.lowRobot=np.array([R1-25,R2-25,R3-25])
                self.highRobot=np.array([R1+25,R2+25,R3+25])

        self.VideoCap=cv2.VideoCapture(1+cv2.CAP_DSHOW)
        ret, frame=self.VideoCap.read()
        self.image=frame


    def finish(self):
        self.VideoCap.release()
        cv2.destroyAllWindows()


    def update(self):
        ret, frame=self.VideoCap.read()
        self.image=frame
        self.imageDraw=frame


    def display(self):
        cv2.imshow('Image Draw', self.imageDraw)


    def robotDetection(self):
        """
        this function detects the robot by detecting the green color in an HSV colorspace 
        then it detects all the contours of the image (after the green mask applied)
        takes the contours with the areas big enough to correspond to the robot (should find 2 
        coutours because there are 2 circles)
        """

        #apply blur filter(reduce noise)
        filter=cv2.blur(self.image, (3, 3))
        if(self.HSVR):
            # Convert image to HSV
            map_hsv = cv2.cvtColor(filter, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(map_hsv, self.lowRobot, self.highRobot)
        else:
            mask=cv2.inRange(filter, self.lowRobot, self.highRobot)

        points=[]
        elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
        surface = 30
        for element in elements:
            if cv2.contourArea(element)>surface:
                ((x, y), rayon)=cv2.minEnclosingCircle(element)
                points.append(np.array([int(x), int(y)]))


        #compute the direction of the robot thanks to the two circles of different sizes
        #then return the position of the robot and the angle (direction)
        if (len(points)>1):
            cv2.circle(self.imageDraw, (points[0][0], points[0][1]), 10, (0, 255, 0), 1)
            direction=[points[1][0]-points[0][0],points[1][1]-points[0][1]]
            theta = np.arctan2(direction[1], direction[0])%(2*np.pi)
            cv2.arrowedLine(self.imageDraw,(points[0][0], points[0][1]),(int(points[0][0]+30*np.cos(theta)),
                            int(points[0][1]+30*np.sin(theta))),color=(0, 255, 0),thickness=1, tipLength=0.2)
            return [points[0][0], points[0][1], theta]
        else:
            return False


    def goalDetection(self):
        """
        this function detects the goal by detecting the red color in an HSV colorpace
        then it detects all the contours of the image (after the red mask applied)
        takes the coutours whose areas are big enough to be the robot 
        (should find 1 coutour corresponding to the red circle)
        """

        #apply blur filter (reduce noise)
        filter=cv2.blur(self.image, (3, 3))
        if(self.HSVG):
            # Convert image to HSV
            map_hsv=cv2.cvtColor(filter, cv2.COLOR_BGR2HSV)
            mask=cv2.inRange(map_hsv, self.lowGoal, self.highGoal)
        else:
            mask=cv2.inRange(filter, self.lowGoal, self.highGoal)

        points=[]
        elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
        surface = 15
        for element in elements:
            if cv2.contourArea(element)>surface:
                ((x, y), rayon)=cv2.minEnclosingCircle(element)
                points.append(np.array([int(x), int(y)]))
        #return the position of the goal
        if (len(points)>0):
            cv2.circle(self.imageDraw, (points[0][0], points[0][1]), 10, (0, 0, 255), 2)
            return [points[0][0], points[0][1]]
        else:
            return False


    def createOccupancyGrid(self,expend):
        """
        This function applies filters on the image to have a good image to analyse
        It returns a binary map that we can give to the function that compute the path
        of the robot
        It also enlarges the obstacles
        """
        # Convert to RGB
        map_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # Blur Image
        map_blur = cv2.blur(map_rgb,(10,10))

        # Apply Bilateral filter
        map_bilateral = cv2.bilateralFilter(map_rgb,5,15,15)

        # Create Binary Image
        map_gray = cv2.cvtColor(map_bilateral, cv2.COLOR_RGB2GRAY)
        _, map_binary = cv2.threshold(map_gray, self.tresh, 255, cv2.THRESH_BINARY_INV)

        # Opening to remove noise
        kernel_morph = np.ones((10,10),np.uint8)
        map_clean = cv2.morphologyEx(map_binary, cv2.MORPH_OPEN, kernel_morph)

        kernel_erode = np.ones((10,10),np.uint8) #60,60
        #dilates to enlarge the obstacles
        kernel_dilate = np.ones((70,70),np.uint8) #140,140
        map_occupancy = cv2.erode(map_clean, kernel_erode, iterations=1)
        if(expend):
            map_occupancy = cv2.dilate(map_occupancy, kernel_dilate, iterations=1)

        return map_occupancy


    def obstaclesDetection(self, expend):
        """
        create the occupancy grid :
        1 = occupied
        0 = free
        """
        map_occupancy=self.createOccupancyGrid(expend)

        width=np.size(map_occupancy,0)
        height=np.size(map_occupancy,1)

        occupancy_grid = np.zeros((width,height))

        for i in range(width):
            for j in range(height):
                result = map_occupancy[i,j]
                if result > 0 :
                    occupancy_grid[i,j] = 1

        return occupancy_grid
