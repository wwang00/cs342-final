from tournament.utils import HACK_DICT
import numpy as np
from numpy.linalg import norm


class HockeyPlayer(object):
    TARGET_SPEED = 15

    def __init__(self, player_id=0):
        self.player_id = player_id
        self.kart = 'wilber'
        self.team = player_id % 2
        self.goal = np.float32([0, 64 if self.team == 0 else -64])

    def act(self, image, player_info):
        """
        :param image: numpy array of shape (300, 400, 3)
        :param player_info: pystk.Player object for the current kart.
        return: Dict describing the action
        """
        puck = np.float32(HACK_DICT['state'].soccer.ball.location)[[0, 2]]
        front = np.float32(player_info.kart.front)[[0, 2]]
        kart = np.float32(player_info.kart.location)[[0, 2]]
        vel = np.float32(player_info.kart.velocity)[[0, 2]]
        speed = norm(vel)

        u = front - kart
        u /= norm(u)

        puck_goal = self.goal - puck
        puck_goal /= norm(puck_goal)
        aim = puck - 1.5 * puck_goal

        v = aim - kart
        v /= norm(v)

        theta = np.arccos(np.dot(u, v))
        signed_theta = -np.sign(np.cross(u, v)) * \
            np.sign(np.dot(u, vel)) * theta

        steer = 10 * signed_theta
        brake = np.degrees(theta) > 30
        acceleration = 1 if speed < self.TARGET_SPEED and not brake else 0

        return {
            'steer': steer,
            'acceleration': acceleration,
            'brake': brake,
            'drift': False,  # np.degrees(theta) > 60,
            'nitro': False,
            'rescue': False
        }
