from tdmclient import ClientAsync, aw
import numpy as np

class LocalNavigator:
    def __init__(self):
        self.client = ClientAsync() # Thymio client
        self.node = aw(self.client.wait_for_node()) # node

        aw(self.node.lock()) # lock node
        aw(self.node.wait_for_variables()) # wait for variables

        self.dist_threshold = 2300 # distance threshold to detect obstacles
        self.motor_speed = 0 # default motor speed
        self.sensor_vals = list(self.node['prox.horizontal']) # sensor values of proxy horizontal
        self.verbose = False # whether to print status message or not
        self.omega = 0 # rotation velocity
        """
        |Degree|Motor speed|velocity (degree/sec)|
        |------|-----------|---------------------|
        | 1080 |    300    |         108         |
        | 554  |    200    |         55          |
        | 370  |    100    |         37          |
        | 325  |    80     |         32          |
        | 222  |    50     |         22          |
        | 135  |    30     |         13          |
        """
        self.is_alter = [] # for checking whether Thymio stuck in deadlock
        self.deadlock_flag = False # whether Thymio stuck in deadlock
        self.turn_direction = 0 # turning direction (left: -1, right: 1, others: 0)
        self.reflected_sensor_vals = list(self.node['prox.ground.reflected']) # sensor values of proxy ground reflected {0...1023}
        self.height_threshold = 70 # threshold recognizing kidnap
        self.kidnap = False # flag of kidnapping
        self.resolved_obstacle = 0 # the number of resolution of obstacle avoidance

    def update_proxsensor(self):
        """
        Update proxy horizontal sensor values.
        """
        self.sensor_vals = list(self.node["prox.horizontal"])

    def motor(self, l_speed=500, r_speed=500):
        """
        param    l_speed: left motor speed (default 500)
        param    r_speed: right motor speed (default 500)
        Set left and right motor speed.
        """
        if self.verbose:
            print("\tSetting speed: ", l_speed, r_speed)
        return {
                "motor.left.target": [l_speed],
                "motor.right.target": [r_speed],
                }

    def print_sensor_values(self):
        """
        Print proxy horizontal sensor values.
        """
        if self.verbose:
            print("\tSensor values (prox_horizontal): ", self.sensor_vals)

    def compute_motor_speed(self):
        """
        Set the motor speed corresponding to the distance to obstacles, and the rotated velocity corresponding to the motor speed.
        Closer obstacles are, less motor speed is set.
        """
        max_sensor_val = max(self.sensor_vals)

        if 0 <= max_sensor_val < 1500:
            self.motor_speed = 200
            self.omega = 55
        elif 1500 <= max_sensor_val < 2000:
            self.motor_speed = 100
            self.omega = 37
        elif 2000 <= max_sensor_val < 2300:
            self.motor_speed = 80
            self.omega = 32
        elif 2300 <= max_sensor_val < 2500:
            self.motor_speed = 50
            self.omega = 22
        elif 2500 <= max_sensor_val:
            self.motor_speed = 30
            self.omega = 13

    async def is_kidnap(self):
        """
        Detect whether Thymio is kidnapped or not.
        """
        await self.client.wait_for_node()
        self.reflected_sensor_vals = list(self.node['prox.ground.reflected']) # update sensor values
        if all([x < self.height_threshold for x in self.reflected_sensor_vals]):
            print(">> Kidnap")
            self.kidnap = True
        else:
            self.kidnap = False

    async def check_deadlock(self):
        """
        Check whether Thymio stuck in deadlock situation or not.
        """
        if len(self.is_alter) > 20 and sum(self.is_alter) == 0: # turn right and turn left alternately over 20 times
            print(">> Deadlock")
            self.deadlock_flag = True

    async def turn_left(self):
        """
        Make Thymio turn left.
        """
        self.turn_direction = -1
        self.omega = self.motor_speed # angle velocity
        self.motor_speed = 0
        await self.node.set_variables(self.motor(-self.omega, self.omega))
        self.is_alter.append(self.turn_direction)

    async def turn_right(self):
        """
        Make Thymio turn right.
        """
        self.turn_direction = 1
        self.omega = -self.motor_speed # angle velocity
        self.motor_speed = 0
        await self.node.set_variables(self.motor(-self.omega, self.omega))
        self.is_alter.append(self.turn_direction)

    async def forward(self, omega=0):
        """
        Make Thymio go forward.
        """
        await self.node.set_variables(self.motor(self.motor_speed-omega, self.motor_speed+omega))
        self.is_alter = [] # reset

    async def backward(self):
        """
        Make Thymio go backward.
        """
        await self.node.set_variables(self.motor(-self.motor_speed, -self.motor_speed))
        self.is_alter = [] # reset

    async def follow_global_path(self, angle):
        """
        param    angle: a given angle to the goal
        Make Thymio follow global path
        """
        self.omega = -int(angle)
        self.motor_speed = 180 - abs(int(3 * self.omega))
        if self.motor_speed < 0: # minimum speed is 0
            self.motor_speed = 0
        if self.omega > 200: # maximum speed is 200
            self.omega = 200
        await self.forward(self.omega)

    async def stop(self):
        """
        Make Thymio stop moving
        """
        self.turn_direction = 0
        self.motor_speed = 0
        await self.node.set_variables(self.motor(self.motor_speed, self.motor_speed))
        self.is_alter = [] # reset

    async def avoid(self, angle):
        """
        param    angle: a given angle to the goal
        Follow the global path if there is no obstacles, and avoid obstacle detected into four cases:
        1. Front obstacles
        2. Left obstacles
        3. Right obstacles
        4. Back obstacles
        """

        # update proxy horizontal sensor values
        self.update_proxsensor()
        front_prox_horizontal = self.sensor_vals[:5] # five front proxy horizontal sensor values
        back_prox_horizontal = self.sensor_vals[5:] # two back proxy horizontal sensor values

        self.print_sensor_values()

        # check whether Thymio is kidnapping
        await self.is_kidnap()

        # compute the proper motor speed according to the distance of obstacle
        self.compute_motor_speed()

        if all([x < self.dist_threshold for x in front_prox_horizontal]): # no obstacle
            if self.resolved_obstacle: # resolved_obstacle > 0
                self.motor_speed = 180
                if self.resolved_obstacle == 1: # for more space
                    if self.turn_direction == 1:
                        await self.turn_right()
                    elif self.turn_direction == -1:
                        await self.turn_left()
                self.resolved_obstacle += 1 # count up
                await self.forward()
                if self.resolved_obstacle > 7:
                    self.resolved_obstacle = 0 # reset
                return
            await self.follow_global_path(angle)
            self.deadlock_flag = False # free to deadlock
            self.is_alter = [] # reset

        if not self.deadlock_flag: # not in deadlock situation
            if front_prox_horizontal[2] > self.dist_threshold: # front obstacle
                if not self.resolved_obstacle:
                    print("Front obstacle")
                self.resolved_obstacle = 1
                if front_prox_horizontal[2] > 4000: # too close to the obstacle
                    await self.backward()

                if (front_prox_horizontal[1] - front_prox_horizontal[3]) < -100: # close to right
                    await self.turn_left()
                elif (front_prox_horizontal[1] - front_prox_horizontal[3]) < 100: # close to left
                    await self.turn_right()
                else:
                    if np.random.randint(20) < 1: # probability 0.05
                        print(">> Explore!")
                        await self.turn_left()
                    else: # probability 0.95
                        await self.turn_right()

            elif any([x > self.dist_threshold for x in front_prox_horizontal[:2]]): # left obstacle
                if not self.resolved_obstacle:
                    print("Left obstacle")
                self.resolved_obstacle = 1
                await self.turn_right()

            elif any([x > self.dist_threshold for x in front_prox_horizontal[3:]]): # right obstacle
                if not self.resolved_obstacle:
                    print("Right obstacle")
                self.resolved_obstacle = 1
                await self.turn_left()

            elif any([x > self.dist_threshold for x in back_prox_horizontal]): # back obstacle
                print("Back obstacle")
                await self.forward()
        else: # in deadlock situation
            await self.turn_right() # turn right until there is no obstacle

        # check whether Thymio stuck in deadlock
        await self.check_deadlock()

    async def run(self, angle):
        """
        param    angle: a given angle to the goal
        Run avoid function making Thymio run.
        """
        self.motor_speed = 0
        self.omega = 0
        await self.avoid(angle)
        return self.motor_speed, self.omega, self.kidnap
