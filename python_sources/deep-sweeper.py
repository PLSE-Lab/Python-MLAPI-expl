#!/usr/bin/env python
# coding: utf-8

# # Set up Tensorboard

# In[ ]:


get_ipython().system('rm -rf /kaggle/working/logs')


# In[ ]:


get_ipython().system('mkdir -p /kaggle/working/logs')
# !tensorboard --logdir=./logs
get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip -o ngrok-stable-linux-amd64.zip')
LOG_DIR = '/kaggle/working/logs' # Here you have to put your log directory
# LOG_DIR = '/kaggle/input/logs' # Here you have to put your log directory
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')


# # Initialize Dependencies

# In[ ]:


get_ipython().system('pip install keras-rl drawSvg hyperbolic')


# In[ ]:


import random
import pickle
import time
import matplotlib.pyplot as plt
from functools import reduce
from collections import deque

import gym
import math

import numpy as np
from gym import error, spaces, utils

from keras import metrics
from keras.callbacks import TensorBoard, BaseLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Activation, Flatten, Conv2D, Activation, BatchNormalization, Input, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import metrics, losses
from rl.core import Processor
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import drawSvg as draw
from drawSvg.widgets import DrawingWidget

get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib notebook


# # Set up MineSweeper

# In[ ]:


class Tile:
    def __init__(self, is_bomb=False):
        self.is_bomb = is_bomb
        self.is_checked = False
        self.is_opened = False
        self.neighbours = []
        self.value = -1

    def check(self):
        self.is_checked = not self.is_checked

    def open(self):
        if not self.is_opened:
            self.is_opened = True
            if self.value == 0:
                for n in self.neighbours: n.open()

    def calculate_value(self):
        self.value = 9 if self.is_bomb else sum(x.is_bomb == 1 for x in self.neighbours)

        return self.value


class Board():
    def __init__(self, width, height, bombs):
        self.width = width
        self.height = height
        self.bombs = bombs
        self.grid = []
        self.generate()

    def generate(self):
        self.__generate_tiles()
        self.__place_bombs()

    def reset(self):
        for tile in self.tiles:
            tile.is_bomb = False
            tile.is_opened = False
            tile.is_checked = False
        self.__place_bombs()

    def __generate_tiles(self):
        for x in range(self.width):
            self.grid.append([])
            for y in range(self.height):
                tile = Tile()
                self.grid[x].append(tile)
                if (x > 0):
                    self.grid[x - 1][y].neighbours.append(tile)
                    tile.neighbours.append(self.grid[x - 1][y])
                    if (y > 0):
                        self.grid[x - 1][y - 1].neighbours.append(tile)
                        tile.neighbours.append(self.grid[x - 1][y - 1])
                    if (y < self.height - 1):
                        self.grid[x - 1][y + 1].neighbours.append(tile)
                        tile.neighbours.append(self.grid[x - 1][y + 1])

                if (y > 0):
                    self.grid[x][y - 1].neighbours.append(tile)
                    tile.neighbours.append(self.grid[x][y - 1])

    def __place_bombs(self):
        bombs_placed = 0 
        
        # Get middle tile
        middle_tile = self.tiles[len(self.tiles) // 2]
        while bombs_placed < self.bombs:
            tile = random.choice(self.tiles)
            # Never place a bomb on another bomb or the middle tile
            if not tile.is_bomb and tile != middle_tile:
                tile.is_bomb = True
                bombs_placed += 1
        for tile in self.tiles:
            tile.calculate_value()

    @property
    def tiles(self):
        return reduce(lambda prev, curr: prev + curr, self.grid)


class Minesweeper():
    def __init__(self, width, height, bombs):
        self.board = Board(width, height, bombs)

    def open(self, x, y):
        self._get_tile(x, y).open()
        
    def is_hail_mary(self, x, y):
        opened_neighbours = len([t for t in self._get_tile(x, y).neighbours if t.is_opened])
        return self.get_score() > 0 and opened_neighbours == 0

    def check(self, x, y):
        self._get_tile(x, y).check()

    def _get_tile(self, x, y):
        return self.board.grid[x][y]

    def reset(self):
        self.board.reset()

    def show_board(self):
        rotated_board = zip(*self.board.grid[::-1])
        for column in rotated_board:
            for tile in column:
                print(tile.value, end='')
            print('')

    def get_score(self):
        return sum(t.is_opened for t in self.board.tiles)

    def is_done(self):
        return self.has_won() or self.has_lost()

    def has_won(self):
        return all(t.is_opened for t in filter(lambda t: not t.is_bomb, self.board.tiles))

    def has_lost(self):
        return any(t.is_opened for t in filter(lambda t: t.is_bomb, self.board.tiles))


# In[ ]:


class MinesweeperEnv(gym.Env):
    FONT_SIZE = 16
    BORDER_WIDTH = 2

    metadata = {'render.modes': ['human']}

    def __init__(self, width, height, bombs):
        self.game = Minesweeper(width, height, bombs)
        self.screen_dimensions = (300, 300)
        self.action_space = spaces.Discrete(width * height)
        self._widget = None
        self._canvas = None
        
        self.steps = 0
        
        self.rewards = {
            'win': 1,
            'loss': -1,
            'valid_move': 0.5,
            'hail_mary': -0.3,
            'invalid_move': -0.5,
        }

        self.observation_space = spaces.Box(0, 1, (width, height, 10), dtype=np.float32)
        self.last_move = (0, 0)
        
        self._memoizable_vars = ['game', 'last_move']

    def step(self, action):
        self.steps += 1

        old_score = self.game.get_score()
        
        # Todo: Check if x and y are not switched around
#         x = action % self.game.board.width
#         y = math.floor(action / self.game.board.width)
        y = action % self.game.board.height
        x = math.floor(action / self.game.board.height)
        
        reward = 0
        applicable_rewards = []
        
        self.last_move = (x, y)
        self.game.open(x, y)
        new_score = self.game.get_score()
        
        if self.game.is_hail_mary(x, y):
            applicable_rewards.append('hail_mary')
        elif new_score > old_score:
            applicable_rewards.append('valid_move')
        else:
            applicable_rewards.append('invalid_move')

        if self.game.has_lost():
            applicable_rewards.append('loss')
        elif self.game.has_won():
            print("FUCKING SOLVED ONE! (in {} steps)".format(self.steps))
            applicable_rewards.append('win')

        terminal = self.game.is_done()
        
        info = {
            "score": self.game.get_score(), 
        }
        
        for reward_name in self.rewards:
            if reward_name in applicable_rewards:
                reward += self.rewards[reward_name]
                info[reward_name] = 1 
            else:
                info[reward_name] = 0
            
        state = self.observe()

        return state, reward, terminal, info

    def observe(self):
#         return np.array([[tile.value if tile.is_opened else -1 for tile in column] for column in self.game.board.grid])
#         values = np.asarray([[tile.value if tile.is_opened else -1 for tile in column] for column in self.game.board.grid])
        width, height, dims = self.observation_space.shape
        values = np.zeros(shape=self.observation_space.shape)
        for y in range(height):
            for x in range(width):
                tile = self.game.board.grid[x][y]
                idx = tile.value if tile.is_opened else 9
                values[x][y][idx] = 1
        return values
    
    def set_memento(self, memento):
        previous_state = pickle.loads(memento)
#         vars(self).clear()
        vars(self).update(previous_state)

    def create_memento(self):
        return pickle.dumps({k : v for k, v in vars(self).items() if k in self._memoizable_vars})
    
#     def set_state(self, state):
#         width, height, dims = self.observation_space.shape
#         for y in range(height):
#             for x in range(width):
#                 tile = self.game.board.grid[x][y]
#                 tile_state = np.argmax(state[x][y])      
#                 tile.is_bomb = False # This method has no way of knowing if a tile is a bomb
#                 if tile_state == 9:
#                     tile.is_opened = False
#                     tile.value = 0
#                 else:
#                     tile.value = tile_state
#                     tile.is_opened = True

    def reset(self):
        self.steps = 0
        self.game.reset()
        # Open middle tile to give the Actor a starting point
        self.game.board.tiles[len(self.game.board.tiles) // 2].is_opened = True
        return self.observe()
    
    @property
    def canvas(self):
        x_blocks = self.game.board.width
        y_blocks = self.game.board.height
        if not self._canvas:
            self._canvas = draw.Drawing(x_blocks, y_blocks, origin=(0,0))
            self._canvas.setRenderSize(500)
        return self._canvas 
    
    @property
    def drawing_widget(self):
        if not self._widget:
            self._widget = DrawingWidget(self.canvas)
            display(self._widget)
        return self._widget

    def render(self, mode='human', close=False):
        if close:
            self._canvas = None
            self._widget = None
            return
        
        self.canvas.clear()
    
        x_blocks = self.game.board.width
        y_blocks = self.game.board.height
        block_width = 32
        block_height = 32

        for x in range(x_blocks):
            for y in range(y_blocks):
                tile = self.game.board.grid[x][y]
                fill_color = 'aqua' if (x + y) % 2 == 1 else 'lightblue' # checkered colored tiles
                fill_color = fill_color if not tile.is_opened else 'lightgrey' # light background for opened tiles 

                if (self.game.has_won()):
                    fill_color = 'lightgreen' # entire board green if won
                elif self.game.has_lost():
                    fill_color = 'red'        # entire board red if lost

                border_color = 'gray' # grey borders around tiles

                last_x, last_y = self.last_move
                if x == last_x and y == last_y:
                    border_color = 'orange' # orange border if tile was the last one to be clicked
                
                self.canvas.append(draw.Rectangle(
                    x, 
                    y, 
                    1, 
                    1, 
                    fill=fill_color,
                    stroke=border_color,
                    stroke_width=0.1,
                ));
                if tile.is_opened:
                    self.canvas.append(draw.Text(str('X' if tile.is_bomb else tile.value),0.4, x + 0.4, y + 0.4, center=0.0, fill='black'))
        self.drawing_widget.refresh()
#                 rect = pygame.Rect(x * block_width,
#                                    y * block_height,
#                                    block_width,
#                                    block_height)
#                 draw.rect(screen, border_color, rect)
#                 draw.rect(screen, fill_color, rect.inflate(-self.BORDER_WIDTH, -self.BORDER_WIDTH))

#                 text = font.render(str('X' if tile.is_bomb else tile.value), True, (0, 0, 0))

#                 screen.blit(text, (x * block_width + (block_width / 2) - (self.FONT_SIZE / 2),
#                                    y * block_height + (block_height / 2) - (self.FONT_SIZE / 2)))
#         pygame.display.flip()


# # AI

# ## Set Up Model Parameters

# In[ ]:


WIDTH = 15
HEIGHT = 12
BOMBS = 30

WIDTH = 30
HEIGHT = 12
BOMBS = 50

# EPISODE_SIZE = 1000
EPISODE_SIZE = WIDTH * HEIGHT * 2
MAX_TRAIN_EPISODES = 1000
TRAINING_STEPS = EPISODE_SIZE * MAX_TRAIN_EPISODES

TRAINING_STEPS = 1000000


TEST_EPISODES = 5

VISUALIZE_TRAINING = False
VISUALIZE_TESTING = False

MEMORY_SIZE = 10000
WARMUP_STEPS = 1000
EPSILON_START = 0.3
EPSILON_END = 0.01
EPSILON_TEST = 0.005
GAMMA = 0.85
TARGET_MODEL_UPDATE = 0.001

LEARNING_RATE=0.001

CONV_COUNT = 5
CONV_FILTER_COUNT = 64


# ## Create MineSweeper Instance

# In[ ]:


env = MinesweeperEnv(WIDTH, HEIGHT, BOMBS)


# ## Build ML Model

# In[ ]:


# in_layer = Input(shape=env.observation_space.shape)
# x = in_layer
in_layer = Input(shape=(None, ) + env.observation_space.shape)
x = Reshape(env.observation_space.shape)(in_layer) # Mitigates the extra dimension added by the SequentialMemory

for _ in range(CONV_COUNT):
    x = Conv2D(CONV_FILTER_COUNT, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

x = Conv2D(1, (1,1))(x)
x = Activation('linear')(x)

out_layer = Flatten()(x)
# out_layer = x

model = Model(inputs=in_layer, outputs=out_layer)
model.summary()

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=LEARNING_RATE))


# In[ ]:


weights_filename = '/kaggle/working/weights.h5f'
checkpoint_weights_filename = '/kaggle/working/checkpoints/weights_{step}.h5f'
get_ipython().system('mkdir -p /kaggle/working/checkpoints/')


# ## Keras-RL DQN Agent

# ### Build Agent

# In[ ]:


memory = SequentialMemory(limit=MEMORY_SIZE, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=EPSILON_START, value_min=EPSILON_END, value_test=EPSILON_TEST, nb_steps=TRAINING_STEPS / 2)
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=-1., value_test=.05, nb_steps=5000)
# policy = BoltzmannQPolicy()

dqn = DQNAgent(
    model=model, 
    nb_actions=env.action_space.n, 
    memory=memory, 
    nb_steps_warmup=WARMUP_STEPS,
    target_model_update=TARGET_MODEL_UPDATE,
    gamma=GAMMA,
    policy=policy,
    enable_double_dqn=True, 
    enable_dueling_network=False, 
    dueling_type='avg',
)
dqn.compile(Adam(lr=LEARNING_RATE), metrics=[metrics.mae])

# dqn.load_weights('/kaggle/input/deep-sweeper/weights.h5f')

# dqn.load_weights(weights_filename)
dqn.save_weights(weights_filename, overwrite=True)


# In[ ]:


get_ipython().system('ls /kaggle/input/deep-sweeper')


# ### Train DQN Agent

# In[ ]:


callbacks = [
    TensorBoard(log_dir='/kaggle/working/logs', batch_size=32, write_graph=True, write_grads=True, write_images=False, update_freq='epoch'),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='auto', cooldown=0, min_lr=0)
    # BaseLogger(),
    # TensorBoard(log_dir='./logs'),
    ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, verbose=0, mode='auto',
#                               cooldown=0, min_lr=0.0001)
]

dqn.fit(
    env,
    nb_steps=TRAINING_STEPS,
    visualize=VISUALIZE_TRAINING,
    verbose=1,
    nb_max_episode_steps=EPISODE_SIZE,
    callbacks=callbacks
)
dqn.save_weights(weights_filename, overwrite=True)


# In[ ]:


dqn.test(env, nb_episodes=TEST_EPISODES * 100, visualize=VISUALIZE_TESTING, nb_max_episode_steps=EPISODE_SIZE)


# In[ ]:


# print("Training...")
# train_results = brain.train(TRAINING_STEPS, EPISODE_SIZE, VISUALIZE_TRAINING)
# test_results = brain.test(5, 1000, VISUALIZE_TESTING)
# brain.save()
# print('Progress saved')


# # Visualize game

# In[ ]:


env = MinesweeperEnv(WIDTH, HEIGHT, BOMBS)

episodes = []

# for _ in range(5):
#     episode = []
#     state = env.reset()
#     while True:
#         action = dqn.forward(state)
#         state, reward, done, info = env.step(action)

#         memento = env.create_memento()
#         episode.append(memento)
        
#         if done:
#             break
#     episodes.append(episode)

# for _ in range(5):
#     episode = []
#     for _ in range(100 * (len(episodes) + 1)):
#         step = {
#             'state': np.random.standard_normal(size=env.observation_space.shape),
#             'prediction': np.random.randint(0, env.action_space.n), 
#             'reward': 1
#         }
#         episode.append(step)
#     episodes.append(episode)

# current_episode = episodes[0]
# current_step = current_episode[0]

widget = env.drawing_widget

current_episode = -1
current_step = -1

prev_btn = widgets.Button(description="Previous step")
next_btn = widgets.Button(description="Next step")
reset_btn = widgets.Button(description="Reset Episode")

step_lbl = widgets.Label(value='Step: ')
reward_lbl = widgets.Label(value='Reward: ')

def do_step():
    global current_step
    current_state = episodes[current_episode][current_step]['state']
    action = dqn.forward(current_state)
    state, reward, done, info = env.step(action)

    memento = env.create_memento()
    episodes[current_episode].append({
        'memento': memento,
        'reward': reward,
        'terminal': done,
        'state': state,
#         'q_values': q_values,
    })
    
q_values_group = draw.Group(opacity=0.5, id='q_values')

def render_q_values():
    q_values_group.children.clear()
#     self.model.predict_on_batch(batch)
    q_values = dqn.model.predict_on_batch(np.array([[env.observe()]]))
    q_values = dqn.compute_q_values(dqn.memory.get_recent_state(env.observe()))
        
#     q_values = q_values.copy() 
    q_values += q_values.min()
    max_q = q_values.max()
    q_values /= max_q if max_q and max_q != 0.0 else 1
        
    for action, value in enumerate(q_values):
#         x = action % env.game.board.width
#         y = math.floor(action / env.game.board.width)
        y = action % env.game.board.height
        x = math.floor(action / env.game.board.height)

        q_values_group.append(draw.Rectangle(
            x, 
            y * value, 
            1, 
            1, 
            fill='limegreen',
            stroke_width=0.1,
        ));
    env.canvas.append(q_values_group)
#     for x in range(env.game.board.width):
#         for y in range(env.game.board.height):
#             tile = self.game.board.grid[x][y]

                

def visualize():
    step = episodes[current_episode][current_step]
    env.set_memento(step['memento'])
    step_lbl.value = 'Step: {}'.format(current_step)
    reward_lbl.value = 'Reward: {}'.format(step['reward'])
    env.render()
    render_q_values()
    
def next_step(b):
    global current_step
    if current_step >= len(episodes[current_episode]) - 1:
        do_step()
    current_step += 1
    visualize()
    
def previous_step(b):
    global current_step
    if current_step > 0:
        current_step -= 1
        visualize()
        
def reset_episode(b):
    global current_episode
    global current_step
    current_episode += 1
    current_step=0
    episodes.append([])
    state = env.reset()
    memento = env.create_memento()
    episodes[current_episode].append({
        'memento': memento,
        'reward': 0,
        'terminal': False,
        'state': state
    })
    visualize()

reset_episode(None)
# interact(visualize_step, 
#          episode=widgets.IntSlider(min=0,max=len(episodes) - 1,step=1,value=0),
#          step=widgets.IntSlider(min=0,max=100 - 1,step=1,value=0)
#         );


prev_btn.on_click(previous_step)
next_btn.on_click(next_step)
reset_btn.on_click(reset_episode)



btn_box = widgets.HBox([prev_btn, next_btn, reset_btn])
lbl_box = widgets.HBox([step_lbl, reward_lbl])

widget_box = widgets.VBox([btn_box, lbl_box])

display(widget_box)

env.render()
widget.refresh()

