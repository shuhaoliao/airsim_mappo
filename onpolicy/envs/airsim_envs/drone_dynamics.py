import airsim
import numpy as  np
import math
from gym import spaces
import configparser


class DroneDynamicsAirsim:
    def __init__(self, cfg, client, id) -> None:
        # config
        self.id = id
        self.name = "cf" + str(100 + id)
        # # AirSim Client
        # self.client = airsim.client.MultirotorClient()
        # self.client.confirmConnection()
        # self.client.enableApiControl(True)
        # self.client.armDisarm(True)
        self.pose_offset = np.array([[0.0, 0.0], [-72.0, -2], [2, -18], [0, -60], [-30, -23], [-72, -23], [-73, -60]])
        self.client = client
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.dt = cfg.getfloat('multirotor', 'dt')
        # start and goal position
        self.start_position = [0, 0, 5]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_distance = 10
        self.goal_random_angle = 0
        self.accept_radius = cfg.getint('environment', 'accept_radius')
        # start_position = [0.0, 0.0, 5.0]
        # goal_position = [28.0, -20.0, 1.0]
        # self.set_start(start_position, random_angle=0)
        # self.set_goal(distance=90, random_angle=0)
        self.goal_rect = None
        self.previous_distance_from_des_point = self.goal_distance
        # states
        self.is_crash = False
        self.x, self.y, self.z = self.get_position()
        self.v_xy = 0
        self.v_z = 0
        self.yaw = self.get_attitude()[2]
        self.yaw_rate = 0
        self.yaw_sp = 0
        self.robot_state = {
            'position': np.zeros(3),
            'linear_velocity': np.zeros(3),
            'attitude': np.zeros(3),
            'angular_velocity': np.zeros(3)
        }
        # cmd
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        self.max_step = 1800
        # action space
        self.acc_xy_max = cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5
        if self.navigation_3d:
            self.state_feature_length = 6
            # self.action_space = spaces.Box(low=np.array([-self.acc_xy_max, -self.v_z_max, -self.yaw_rate_max_rad]),
            #                                high=np.array([self.acc_xy_max, self.v_z_max, self.yaw_rate_max_rad]),
            #                                dtype=np.float32)
            self.action_space = spaces.MultiDiscrete([11, 11])
        else:
            self.state_feature_length = 4
            self.action_space = spaces.MultiDiscrete([11])
            # self.action_space = spaces.Box(low=np.array([-self.acc_xy_max, -self.yaw_rate_max_rad]),
            #                                high=np.array([self.acc_xy_max, self.yaw_rate_max_rad]),
            #                                dtype=np.float32)

    def reset(self, yaw_degree, sample_area):
        # self.client.reset()
        # reset goal
        # self.update_goal_pose()
        pose = self.client.simGetObjectPose(self.name)
        pose.position.x_val = self.pose_offset[sample_area][0]
        pose.position.y_val = self.pose_offset[sample_area][1]
        yaw_noise = math.pi * 2 * np.random.random()
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.name)
        self.client.enableApiControl(True, vehicle_name=self.name)
        self.client.armDisarm(True, vehicle_name=self.name)

        self.is_crash = False
        # reset start
        # yaw_noise = self.start_random_angle * np.random.random()

        # set airsim pose
        # pose = self.client.simGetObjectPose(self.name)
        # pose.position.x_val = self.start_position[0]
        # pose.position.y_val = self.start_position[1]
        # pose.position.z_val = - self.start_position[2]
        # pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        # self.client.simSetVehiclePose(pose, True, vehicle_name=self.name)

        # self.client.simPause(False)
        self.step = 0

        # take off
        # self.client.moveToZAsync(-self.start_position[2], 2, vehicle_name=self.name)
        f = self.client.moveByRollPitchYawZAsync(0, 0, np.radians(yaw_degree), -self.start_position[2], 2, vehicle_name=self.name)
        # self.client.simPause(True)
        pre_collision_info = self.client.simGetCollisionInfo(vehicle_name=self.name)
        self.goal_distance = self.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.goal_distance
        return f

    def update_goal_pose(self):
        # if goal is given by rectangular mode
        if self.goal_rect is None:
            distance = self.goal_distance
            noise = np.random.random()
            angle = noise * self.goal_random_angle  # (0~2pi)
            goal_x = distance * math.cos(angle) + self.start_position[0]
            goal_y = distance * math.sin(angle) + self.start_position[1]
        else:
            goal_x, goal_y = self.get_goal_from_rect(self.goal_rect, self.goal_random_angle)
            self.goal_distance = math.sqrt(goal_x * goal_x + goal_y * goal_y)
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]

        print('New goal pose: ', self.goal_position)

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle

    def set_goal(self, distance=None, random_angle=0, rect=None):
        if distance is not None:
            self.goal_distance = distance
        self.goal_random_angle = random_angle
        if rect is not None:
            self.goal_rect = rect

    def get_goal_from_rect(self, rect_set, random_angle_set):
        rect = rect_set
        random_angle = random_angle_set
        noise = np.random.random()  # (-0.5~0.5)
        angle = random_angle * noise - math.pi  # -pi~pi
        rect = [-128, -128, 128, 128]
        # goal_x = 100*math.sin(angle)
        # goal_y = 100*math.cos(angle)

        if abs(angle) == math.pi / 2:
            goal_x = 0
            if angle > 0:
                goal_y = rect[3]
            else:
                goal_y = rect[1]
        if abs(angle) <= math.pi / 4:
            goal_x = rect[2]
            goal_y = goal_x * math.tan(angle)
        elif abs(angle) > math.pi / 4 and abs(angle) <= math.pi / 4 * 3:
            if angle > 0:
                goal_y = rect[3]
                goal_x = goal_y / math.tan(angle)
            else:
                goal_y = rect[1]
                goal_x = goal_y / math.tan(angle)
        else:
            goal_x = rect[0]
            goal_y = goal_x * math.tan(angle)

        return goal_x, goal_y

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm
        @param {type}
        @return: state_norm
                    normalized state range 0-255
        '''

        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi
        relative_pose_z = self.get_position()[2] - self.goal_position[2]  # current position z is positive
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) * 255

        # current speed and angular speed
        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]
        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) * 255
        linear_velocity_z = velocity[1]
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) * 255
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) * 255

        if self.navigation_3d:
            # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
            self.state_raw = np.array(
                [distance, relative_pose_z, math.degrees(relative_yaw), linear_velocity_xy, linear_velocity_z,
                 math.degrees(velocity[2])])
            state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm,
                                   linear_velocity_z_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm
        else:
            self.state_raw = np.array(
                [distance, math.degrees(relative_yaw), linear_velocity_xy, math.degrees(velocity[2])])
            state_norm = np.array([distance_norm, relative_yaw_norm, linear_velocity_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm

        return state_norm

    def _get_relative_yaw(self):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type}
        @return:
        '''
        current_position = self.get_position()
        # get relative angle
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.get_attitude()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def get_position(self):
        position = self.client.simGetObjectPose(self.name).position
        self.x = position.x_val
        self.y = position.y_val
        self.z = - position.z_val
        return [position.x_val, position.y_val, -position.z_val]

    def get_velocity(self):
        states = self.client.getMultirotorState()
        linear_velocity = states.kinematics_estimated.linear_velocity
        angular_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2))
        velocity_z = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose(self.name).orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_attitude_cmd(self):
        return [0.0, 0.0, self.yaw_sp]

    def get_distance_to_goal_2d(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(
            self.get_position()[1] - self.goal_position[1], 2))

    def get_distance_to_goal_3d(self):
        current_pose = np.array(self.get_position())
        relative_pose = current_pose - self.goal_position
        return np.sqrt(np.sum(relative_pose ** 2))

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance_to_goal_2d() < self.accept_radius:
            in_desired_pose = True
        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            is_crashed = True
        return is_crashed

if __name__ == '__main__':
    config_file = 'cfg/default.cfg'
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    c = airsim.client.MultirotorClient('127.0.0.1')
    a = DroneDynamicsAirsim(cfg, c, 1)