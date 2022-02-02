# MobileRobotics

The goal of this project is to make a thymio robot move in an environment with obstacle to reach a goal position. 

## Task1 : Create the environment 

Your environment has to contain a set of obstacles that the Thymio avoids through global navigation.
That is to say, the Thymio should avoid these obstacles without using the sensors to detect them. 

## Task 2 : Global Navigation

The objective is that the Thymio goes from an arbitrary position in the map to a target that can be placed
anywhere in the environment. These will be changed during the demo to see how your system performs. 

## Task 3 : Motion Control & Filtering

You will have to control the robot to help it move along the path. This requires an accurate estimate of the
position of the robot which you will have to obtain through filtering to compensate for noisy or lossy data. 

## Task 4 : Avoid Obstacle 

While navigating, the Thymio will have to use local navigation to avoid physical obstacles that can be put
in its path at any point in time. You are free to choose what these physical objects are.
