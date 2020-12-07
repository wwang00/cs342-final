# from tournament.utils import HACK_DICT
import numpy as np
from numpy.linalg import norm
import oracle_agent.utils as utils
import sys
# sys.path.insert(1, './solution')
# import models
from oracle_agent.models import Detector
import torch
from os import path
import torch


class HockeyPlayer(object):
    TARGET_SPEED = 25
    DRIFT_ANGLE = 20
    BRAKE_ANGLE = 20
    DEFENSE_RADIUS = 40
    ZOOM_TIME = 30

    PUCK = None
    PUCK_T = 0

    def __init__(self, player_id=0, kart='tux'):
        # constants
        self.player_id = player_id
        self.kart = kart
        self.team = player_id % 2
        self.own_goal = np.float32([0, -65 if self.team == 0 else 65])
        self.goal = np.float32([0, 64 if self.team == 0 else -64])
        self.model = Detector()
        self.model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
        # self.model.load_state_dict(torch.load('det.th'))
        # self.model = self.model.to(torch.device('cuda'))
        self.model = self.model.cuda()

        self.model.eval()

        # states
        self.offense = (player_id >= 2)
        self.pucklock = True

        # vars
        self.puck = np.float32([0, 0])
        self.t = 0
        self.delta_puck = np.float32([0, 0])
        self.last_seen = 0

        # recurrent info
        self.last_puck =  np.float32([0, 0])
        self.last_loc = None 

    def act(self, image, player_info, game_state=None, mask=None):
        """
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        dets, depth, is_puck = self.model.detect(torch.from_numpy(image/255.0).float().permute(2, 0, 1).cuda())
        # dets, depth, is_puck = self.model.detect(torch.from_numpy(image/255.0).float().permute(2, 0, 1))


        front = np.float32(player_info.kart.front)[[0, 2]]
        kart = np.float32(player_info.kart.location)[[0, 2]]
        vel = np.float32(player_info.kart.velocity)[[0, 2]]
        speed = norm(vel)

        if self.last_loc is None or float(norm(kart - self.last_loc)) > 10:
            self.t = 0
        self.t+=1

        self.last_loc = kart

        coords = dets[1][0]
        proj =  np.array(player_info.camera.projection).T @ np.array(player_info.camera.view).T
        puck = utils.center_to_world(coords[1], coords[2], 400, 300, proj)

        # if we see puck
        if puck is not None and dets[1][0][0] > 2 and is_puck > 2:
            # draw_cords(image, puck, proj)
            puck = puck[[0,2]]
            self.delta_puck = (puck - self.last_puck) / (self.t - self.last_seen)
            self.last_puck = puck
            self.last_seen = self.t
            self.pucklock = True

        # if we recently saw puck
        elif abs(self.last_seen - self.t) < 5:
            puck = self.last_puck
        else:
            # we lost puck :(
            puck = np.float32([0, 0])


        # draw_arr(image, np.array([puck is not None])) # do we see puck
        # draw_arr(image, np.array([self.t]), ht=50) # time step
        # draw_arr(image, np.array([dets[1][0][0], is_puck]), ht=70) # confidence



        if self.offense and self.t < self.ZOOM_TIME:
            puck = np.float32([0,0])

            
            
        u = front - kart
        u /= norm(u)

        # find aimpoint
        if self.offense:
            puck_goal = self.goal - puck
            puck_goal /= norm(puck_goal)
            kart_puck_dist = norm(puck - kart)
            aim = puck - puck_goal * kart_puck_dist / 2
            if not self.pucklock:
                aim = self.own_goal
        else:  # defense
            own_goal_puck = puck - self.own_goal
            own_goal_puck_dist = norm(own_goal_puck)

            if own_goal_puck_dist < self.DEFENSE_RADIUS:
                aim = puck - 1 * own_goal_puck / own_goal_puck_dist
            elif np.abs(kart[1]) < np.abs(self.own_goal[1]):
                aim = self.own_goal
                u = -u
                vel = -vel
            else:
                aim = self.goal

        # draw_cords(image, np.array([aim[0], .3, aim[1]]), proj)

        # get steer angles
        v = aim - kart
        v /= norm(v)
        theta = np.degrees(np.arccos(np.dot(u, v)))
        signed_theta = -np.sign(np.cross(u, v)) * (.4 + np.sign(np.dot(u, vel)) * theta / 8)
        steer = signed_theta*1.0
        # brake or accelerate
        drift = False
        if self.offense:
            if self.t < self.ZOOM_TIME:
                brake = False
                drift = False
                accel = 1
            elif self.pucklock:
                brake = self.BRAKE_ANGLE <= theta
                drift = self.DRIFT_ANGLE <= theta and not brake
                accel = 1 if speed < self.TARGET_SPEED and not brake else 0

            else:
                brake = True
                drift = False
                accel = 0

                # steer = -1
        else:  # defense
            if own_goal_puck_dist < self.DEFENSE_RADIUS:
                brake = self.BRAKE_ANGLE <= theta
                drift = self.DRIFT_ANGLE <= theta and not brake
                accel = 1 if speed < self.TARGET_SPEED and not brake else 0
            elif np.abs(kart[1]) < np.abs(self.own_goal[1]):
                brake = True
                accel = 0
            else:
                brake = (np.sign(np.dot(u, vel)) == 1 and speed > 2) or 5 <= theta
                accel = 1 if brake == 0 and np.sign(np.dot(u, vel)) == -1 and speed > 2 else 0

        return {
            'steer': steer,
            'acceleration': accel,
            'brake': brake,
            'drift': False,
            'nitro': False,
            'rescue': False,
            'fire': True
        }