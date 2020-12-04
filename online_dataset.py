import random
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pystk
from oracle_agent.player import HockeyPlayer
from tournament.play import DummyPlayer
from tournament.utils import Player
import numpy as np

# based on Tournament, used so we can throw out stuff we dont need

class HockeyDataset(IterableDataset):
    _singleton = None

    def __init__(self, max_frames=1200, transform=lambda x,y: (x,y), screen_dim=(150,200)):
        assert HockeyDataset._singleton is None, "Cannot create more than one Tournament object"
        HockeyDataset._singleton = self

        self.graphics_config = pystk.GraphicsConfig.hd()
        self.graphics_config.screen_width = screen_dim[1]
        self.graphics_config.screen_height = screen_dim[0]
        pystk.init(self.graphics_config)

        self.max_frames = max_frames
        self.transform = transform
        self.karts = ['adiumy', 'amanda', 'beastie', 'emule', 'gavroche',
                'gnu', 'hexley', 'kiki', 'konqi', 'nolok', 'pidgin',
                'puffy', 'sara_the_racer', 'sara_the_wizard',
                'suzanne', 'tux', 'wilber', 'xue']

    # no side effects
    def make_race(self, pp_team):
        race_config = pystk.RaceConfig(
            num_kart=pp_team*2, 
            track='icy_soccer_field',
            mode=pystk.RaceConfig.RaceMode.SOCCER
        )
        race_config.players.pop()

        players = list()
        for i in range(4):
            if i % 2 == 0:
                players.append(Player(HockeyPlayer(player_id=i, kart=random.choice(self.karts)), i % 2))
            else:
                players.append(DummyPlayer(i % 2))

        for p in players:
            race_config.players.append(p.config)

        return players, pystk.Race(race_config)

    # reset race, make a random race
    def __iter__(self):
        self.players, self.k = self.make_race(2)
        self.k.start()
        self.k.step()

        return self.gen()

    def random_ball(self):
        return [random.uniform(-20, 20), 0.5, random.uniform(-40, 40)]

    def gen(self):
        try:
            state = pystk.WorldState()
            state.update()
            for t in range(self.max_frames):
            # for t in range(self.max_frames):

                list_actions = []

                #for each character, fill in the hacked data, play the character
                for i, p in enumerate(self.players):
                    # render_data -> image, instance/semantic segmentation, depth


                    image = np.array(self.k.render_data[i].image)
                    mask = (self.k.render_data[i].instance == 134217729)
                    

                    player = state.players[i]
                    action = pystk.Action()

                    # play this player
                    for k, v in p(image, player, game_state=state, mask=mask).items():
                        setattr(action, k, v)

                    # yielding after actions allows for manipulation of image by controller!
                    # yield image, mask, state
                    yield self.transform(image, mask)

                    list_actions.append(action)

                # Game over.
                if not self.k.step(list_actions):
                    break
                state.update()
                if state.soccer.ball.location[1] > 5: #the game likes to drop it from the sky
                    state.set_ball_location(self.random_ball())


        except Exception as e:
            print('error:' + str(e))
        finally:
            self.k.stop()
            self.k = None
        raise StopIteration


if __name__ == "__main__":
    iter = 0
    for image, mask in HockeyDataset():
        print(image.shape, mask.shape)
        if iter == 5:
            break
        iter+=1
    print('ok')