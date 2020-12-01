import pystk
import numpy as np
import torch


# Privileged information.
HACK_DICT = dict()
HM_RADIUS = 2


class Player:
    def __init__(self, player, team=0):
        self.player = player
        self.team = team

    @property
    def config(self):
        return pystk.PlayerConfig(
            controller=pystk.PlayerConfig.Controller.PLAYER_CONTROL,
            kart=self.player.kart,
            team=self.team
        )

    def __call__(self, image, player_info):
        return self.player.act(image, player_info)


class Tournament:
    _singleton = None

    def __init__(self, players, screen_width=400, screen_height=300, track='icy_soccer_field'):
        assert Tournament._singleton is None, "Cannot create more than one Tournament object"
        Tournament._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_width
        self.graphics_config.screen_height = screen_height
        pystk.init(self.graphics_config)

        self.race_config = pystk.RaceConfig(num_kart=len(
            players), track=track, mode=pystk.RaceConfig.RaceMode.SOCCER)
        self.race_config.players.pop()

        self.active_players = []
        for p in players:
            if p is not None:
                self.race_config.players.append(p.config)
                self.active_players.append(p)

        self.k = pystk.Race(self.race_config)

        self.k.start()
        self.k.step()

    def play(self, save=None, max_frames=50):
        state = pystk.WorldState()

        if save is not None:
            import PIL.Image
            import PIL.ImageDraw
            import PIL.ImageFont
            import os
            import torchvision.transforms.functional as T
            if not os.path.exists(save):
                os.makedirs(save)

        for t in range(max_frames):
            print('\rframe %d' % t, end='\r')

            state.update()
            if(state.soccer.ball.location[1] > 5):
                state.set_ball_location([np.random.uniform(-20, 20), 0, np.random.uniform(-20, 20)])
                state.update()

            list_actions = []
            for i, p in enumerate(self.active_players):
                HACK_DICT['render_data'] = self.k.render_data[i]
                HACK_DICT['state'] = state

                player = state.players[i]
                image = self.k.render_data[i].image

                action = pystk.Action()
                player_action = p(image, player)
                for a in player_action:
                    setattr(action, a, player_action[a])

                list_actions.append(action)

                if save is not None:
                    image = np.array(T.resize(PIL.Image.fromarray(image), (150, 200)))[54:]
                    heatmap = np.zeros((96, 200), dtype=float)
                    puck_mask = np.array(self.k.render_data[i].instance[108:] == 134217729)
                    puck_visible = np.any(puck_mask)
                    if puck_visible:
                        puck_loc = np.argwhere(puck_mask).mean(0) / 2
                        y, x = np.arange(96, dtype=float), np.arange(200, dtype=float)
                        gy = np.exp(-((y - puck_loc[0]) / HM_RADIUS)**2)
                        gx = np.exp(-((x - puck_loc[1]) / HM_RADIUS)**2)
                        heatmap = gy[:, None] * gx[None]
                        heatmap = np.round(heatmap / np.max(heatmap), 4)

                    # save to disk
                    image = PIL.Image.fromarray(image)
                    heatmap = torch.from_numpy(heatmap)
                    # font = PIL.ImageFont.truetype("font.ttf", 16)
                    # draw = PIL.ImageDraw.Draw(image)
                    # loc_x = round(state.karts[i].location[0], 2)
                    # loc_z = round(state.karts[i].location[2], 2)
                    # draw.text((0, 20), f'T: {state.time}', (255, 0, 0), font)
                    # draw.text((0, 40), f'({loc_x}, {loc_z})', (255, 0, 0), font)
                    image.save(os.path.join(save, 'player%02d_%05d.png' % (i, t)))
                    torch.save(heatmap, os.path.join(save, 'player%02d_%05d.pt' % (i, t)))

            s = self.k.step(list_actions)
            if not s:  # Game over
                break

        if save is not None:
            import subprocess
            for i, p in enumerate(self.active_players):
                dest = os.path.join(save, 'player%02d' % i)
                output = save + '_player%02d.mp4' % i
                subprocess.call([
                    'ffmpeg', '-y', '-framerate', '10', '-i',
                    dest + '_%05d.png', '-pix_fmt', 'yuv420p', output
                ])
        if hasattr(state, 'soccer'):
            return state.soccer.score
        return state.soccer_score

    def close(self):
        self.k.stop()
        del self.k
