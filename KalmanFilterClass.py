import numpy as np

class KalmanFilterClass(object):
    def __init__(self):

        # State vector : the robot position and orientation (x, y, alpha)
        self.x=np.array([[0],
                          [0],
                          [0]])

        # input vector : the velocity of the robot (x_dot, y_dot, omega)
        self.u=np.array([[0],
                          [0],
                          [0]])

        # Matrix of system Dynamics
        self.A=np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        # Matrix of input dynamics (timestep dependent)
        self.B=np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        # Observation matrix, robot position and orientation can be observed
        self.H=np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])

        # state dynamics noise covariance matrix
        self.Q=np.array([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10]])

        # measurement noise covariance matrix
        self.R=np.array([[10, 0, 0],
                          [0, 10, 0],
                          [0, 0, 10]])

        self.P=np.eye(self.A.shape[1])

    def setState(self, point):
        """
        initialize the kalman filter with the first measured robot position
        and orientation
        """
        self.x=np.array([[point[0]],
                          [point[1]],
                          [point[2]]])


    def predict(self, input, timeStep):
        """
        Predict the next state
        use the previous state and the input (thymio odometry) to predict the
        next state
        update the uncertainty matrix
        return the predicted state (robot position and orientation)
        """
        self.u = np.array([[input[0]],[input[1]],[input[2]]])
        self.x=np.dot(self.A, self.x) + np.dot((timeStep*self.B), self.u)
        # Calcul de la covariance de l'erreur
        self.P=np.dot(np.dot(self.A, self.P), self.A.T)+self.Q
        return [self.x[0,0],self.x[1,0],self.x[2,0]]


    def update(self, meas):
        """
        Use the measurement to calculate the kalman gain and make a correction
        to the predicted state
        update the uncertainty matrix
        return the updated state (robot position and orientation)
        """
        z = np.array([[meas[0]],[meas[1]],[meas[2]]])

        #  Kalman gain computation
        S=np.dot(self.H, np.dot(self.P, self.H.T))+self.R
        K=np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # Correction / innovation
        self.x=self.x+np.dot(K, (z-np.dot(self.H, self.x)))
        I=np.eye(self.H.shape[1])
        self.P=(I-(K*self.H))*self.P

        return [self.x[0,0],self.x[1,0],self.x[2,0]]
