from tournament.utils import HACK_DICT
import numpy as np
from numpy.linalg import norm
import oracle_agent.utils as utils
import sys
sys.path.insert(1, './solution')
import models
import torch
from os import path


class HockeyPlayer(object):
    TARGET_SPEED = 15
    DRIFT_ANGLE = 20
    BRAKE_ANGLE = 30
    DEFENSE_RADIUS = 40

    def __init__(self, player_id=0, kart='wilber'):
        self.player_id = player_id
        self.kart = kart
        self.team = player_id % 2
        self.offense = True #not player_id < 2
        self.goal = np.float32([0, 64 if self.team == 0 else -64])
        self.model = models.Detector()
        self.model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), "..", path.join('solution', 'det.th')), map_location='cpu'))
        # self.model = self.model.cuda()
        self.own_goal = np.float32([0, -65 if self.team == 0 else 65])
        self.last_puck = None

    def act(self, image, player_info, game_state=None, mask=None):
        """
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        # puck = np.float32(HACK_DICT['state'].soccer.ball.location)[[0, 2]]
        # dets, depth, is_puck = self.model.detect(torch.from_numpy(image/255.0).float().permute(2, 0, 1).cuda())
        dets, depth, is_puck = self.model.detect(torch.from_numpy(image/255.0).float().permute(2, 0, 1))

        puck = dets[1][0]
        #print(puck[1]/400)
        #print(puck[2]/300)
        puck_updated = False
        if is_puck > 0 and puck[0] > 2 and len(dets[0]) > 0 and abs(puck[1] - dets[0][0][1]) + abs(puck[2] - dets[0][0][2]) > 45: # and abs(puck[3] - dets[0][0][3]) + abs(puck[4] - dets[0][0][4]) > 10:
          puck = utils.center_to_world(puck[1], puck[2], 400, 300, np.array(player_info.camera.projection).T @ np.array(player_info.camera.view).T)
          if puck is not None and np.linalg.norm(puck - player_info.kart.location) > 2:
            self.last_puck = puck
            puck_updated = True
            puck = puck[[0, 2]]
          else:
            puck = self.last_puck
            if puck is not None:
              puck = puck[[0, 2]]
            else:
              puck = np.array([0, 0])
        else:
          puck = self.last_puck
          if puck is None:
            puck = np.array([0, 0]) # estimate at center before known
          else:
            puck = puck[[0, 2]]
        realpuck = np.float32(game_state.soccer.ball.location)[[0, 2]]
        puck_map = np.zeros((66 * 4, 66 * 4, 3))
        puck_map[int(round(-66 + 66 * 2 - 1)):int(round(-66 + 66 * 2 + 133)), int(round(-50 + 66 * 2 - 1)):int(round(-50 + 66 * 2 + 1))] = [100, 0, 0]
        puck_map[int(round(-66 + 66 * 2 - 1)):int(round(-66 + 66 * 2 + 133)), int(round(-50 + 66 * 2 - 1 + 100)):int(round(-50 + 66 * 2 + 101))] = [100, 0, 0]
        puck_map[int(round(-66 + 66 * 2 - 1)):int(round(-66 + 66 * 2 + 1)), int(round(-50 + 66 * 2 - 1)):int(round(50 + 66 * 2 + 1))] = [100, 0, 0]
        puck_map[int(round(66 + 66 * 2 - 1)):int(round(66 + 66 * 2 + 1)), int(round(-50 + 66 * 2 - 1)):int(round(50 + 66 * 2 + 1))] = [100, 0, 0]
        puck_map[int(round(puck[1] + 66 * 2 - 2)):int(round(puck[1] + 66 * 2 + 2)), int(round(puck[0] + 66 * 2 - 2)):int(round(puck[0] + 66 * 2 + 2))] += [255, 0, 0]
        puck_map[int(round(realpuck[1] + 66 * 2 - 2)):int(round(realpuck[1] + 66 * 2 + 2)), int(round(realpuck[0] + 66 * 2 - 2)):int(round(realpuck[0] + 66 * 2 + 2))] += [0, 255, 0]
        front = np.float32(player_info.kart.front)[[0, 2]]
        kart = np.float32(player_info.kart.location)[[0, 2]]
        puck_map[int(round(kart[1] + 66 * 2 - 2)):int(round(kart[1] + 66 * 2 + 2)), int(round(kart[0] + 66 * 2 - 2)):int(round(kart[0] + 66 * 2 + 2))] += [0, 0, 255]
        vel = np.float32(player_info.kart.velocity)[[0, 2]]
        speed = norm(vel)

        u = front - kart
        u /= norm(u)

        # find aimpoint
        if self.offense:
            puck_goal = self.goal - puck
            puck_goal /= norm(puck_goal)
            kart_puck_dist = norm(puck - kart)
            aim = puck - puck_goal * kart_puck_dist / 2
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

        # get steer angles
        v = aim - kart
        v /= norm(v)
        theta = np.degrees(np.arccos(np.dot(u, v)))
        signed_theta = -np.sign(np.cross(u, v)) * np.sign(np.dot(u, vel)) * theta
        steer = signed_theta / 8

        # brake or accelerate
        drift = False
        if self.offense:
            brake = self.BRAKE_ANGLE <= theta
            drift = self.DRIFT_ANGLE <= theta and not brake
            accel = 1 if speed < self.TARGET_SPEED and not brake else 0
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
            'puck_map': puck_map
        }