#!/usr/bin/env python
# coding: utf-8

# # Solving OpenAI Gym's Mountain Car Continuous Control task using Deep Deterministic Policy Gradients  
# 
# This notebook uses a modified version of [Udacity's DDPG model](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) to solve OpenAI Gym's [MountainCarContinuous-v0](https://github.com/openai/gym/wiki/MountainCarContinuous-v0) continuous control problem using [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) as part of the [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t) quadcopter project. 
# 
# The code is available on my github repo at https://github.com/samhiatt/ddpg_agent. This kernel version uses code from [this commit](https://github.com/samhiatt/ddpg_agent/tree/6131f08d4a4244e8a9bf5cad430774d3762e33d5).
# 
# Solving the MountainCarContinuous problem with DDPG is a particularly good place to start as its 2-dimensional continuous state space (position and velocity) and 1-dimensional continuous action space (forward, backward) are easy to visualize in two dimensions, lending to an intuitive understanding of hyperparameter tuning. 
# 
# Andre Muta's [DDPG-MountainCarContinuous-v0](https://github.com/amuta/DDPG-MountainCarContinuous-v0) repo was helpful in suggesting some good visualizations as well as giving some good hyperparameters to start with. It looks like he uses the same code from the nanodegree quadcopter project and uses it to solve the MountainCarContinuous problem as well. His [plot_Q method in MountainCar.py](https://github.com/amuta/DDPG-MountainCarContinuous-v0/blob/master/MountainCar.py) was particularly helpful by showing how to plot Q_max, Q_std, Action at Q_max, and Policy. Adding a visualization of the policy gradients and animating the training process ended up helping me better understand the problem and the effects of various hypterparemeters. 
# 
# See [kernel version 61](https://www.kaggle.com/samhiatt/mountaincarcontinuous-v0-ddpg?scriptVersionId=15941050) for a good solution with and animation of policy/value functions and episode states.
# 
# [Kernel version 72](https://www.kaggle.com/samhiatt/mountaincarcontinuous-v0-ddpg?scriptVersionId=16046023) uses batch norm _before_ the critic's last fully-connected layer, as opposed to after it as in earlier versions. This seems to have the effect of reducing the magnitude of the actor's actions which seems to end up achieving a higher test score.
# 
# Using action repeat seemes to help speed up training, and also seems to work better during initial exploratory episodes.
# 
# Scaling rewards with np.log1p(reward) seems to help speed up training significantly. Using batch normalization before the Critic's final layer seems to have a similar effect. 
# 
# Note that subsequent runs of [Version 11](https://www.kaggle.com/samhiatt/ddpg-mountaincarcontinuous-v0?scriptVersionId=15205346) with the same hyperparameters did not find a workable policy and would usually end up with a policy always pushing forward (action=1). 
# 
# [Version 19](https://www.kaggle.com/samhiatt/ddpg-mountaincarcontinuous-v0?scriptVersionId=15555111) appears to be stable, finding a workable policy after about 60 episodes.
# 
# ### This kernel version includes the following:
# * All hyperparameters configurable in `DDPG.__init__`
# * mp4 output of training animation using ffmpeg binary from imagio-ffmpeg 
# * re-worked architecture for tracking training history.
# * Batch Normalization _before_ Critic's last dense layer.
# * Batch Normalization before Actor's last dense layer. (disabled)
# * ~~L2 Regularization on each dense layer~~
# * Action repeat (n=5)
# * ~~LeakyReLUs~~
# * Input normalization preprocessing
# * Half of the nodes per layer compared to Udacity example model.
# * Epsilon decay after each episode. 
# * Tests policy with epsilon=0 after each episode.
# * ~~Batch Normalization~~
# * ~~Tries to select a training batch that includes at least one positive reward.~~
# * ~~Option to only remember episodes where the total reward is positive. (Set to False in this version)~~
# 
# ### More things to try:
# * **Use `env.goal_position` to enhance reward function**
# * Using BatchNorm on all hidden layers
# * Write training history to file.
# * Snapshot model weights while training.
# * Alternative methods for decaying epsilon, maybe linked to test score / training score trends?
#     * Modulate epsilon depending on latest test scores.
#         * Decrease action_repeat after repeated high test scores.
# 
# ## Credits
# * [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
# * Andre Muta's [DDPG-MountainCarContinuous-v0](https://github.com/amuta/DDPG-MountainCarContinuous-v0).
# * Thanks to [Eli Bendersky](https://eli.thegreenplace.net/2016/drawing-animated-gifs-with-matplotlib/) for help with matplotlib animations. 
# * Thanks to [Joseph Long](https://joseph-long.com/writing/colorbars/) for help with matplotlib colorbar axes placement.

# ### Visualization TODOs:
# * Print hyperparameters.
# * Chart epsilon, action_repeat, batch_size / memory_size / buffer_length.
# * Plot last n memory samples used training.
# * ~~Make one animation with all training steps~~
# * ~~Plot training/test scores~~
# * ~~Plot current episode~~
# * ~~Plot last test episode~~

# In[ ]:


import gym
import numpy as np
import warnings

warnings.simplefilter('ignore')

env = gym.make('MountainCarContinuous-v0')
print('Continuous action space: (%.3f to %.3f)'%(env.action_space.low, env.action_space.high))
print('Reward range: %s'%(str(env.reward_range)))
for i in range(len(env.observation_space.low)):
    print('Observation range, dimension %i: (%.3f to %.3f)'%
          (i,env.observation_space.low[i], env.observation_space.high[i]))


# In[ ]:


import numpy as np
import copy
import random
import sys
from collections import namedtuple, deque

from keras import backend as K
from keras import layers, models, optimizers, regularizers

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, env, train_during_episode=True,
                 discount_factor=.999,
                 tau_actor=.2, tau_critic=.2,
                 lr_actor=.0001, lr_critic=.005,
                 bn_momentum_actor=.9, bn_momentum_critic=.9,
                 ou_mu=0, ou_theta=.1, ou_sigma=1,
                 activation_fn_actor='sigmoid',
                 replay_buffer_size=10000, replay_batch_size=64,
                 l2_reg_actor=.01, l2_reg_critic=.01,
                 relu_alpha_actor=.01, relu_alpha_critic=.01,
                 dropout_actor=0, dropout_critic=0,
                 hidden_layer_sizes_actor=[32,64,32],
                 hidden_layer_sizes_critic=[[32,64],[32,64]], ):

        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.train_during_episode = train_during_episode

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low,
                self.action_high, activation_fn=activation_fn_actor, relu_alpha=relu_alpha_actor,
                bn_momentum=bn_momentum_actor, learn_rate=lr_actor, l2_reg=l2_reg_actor,
                dropout=dropout_actor, hidden_layer_sizes=hidden_layer_sizes_actor, )
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low,
                self.action_high, activation_fn=activation_fn_actor, relu_alpha=relu_alpha_actor,
                bn_momentum=bn_momentum_actor, learn_rate=lr_actor, l2_reg=l2_reg_actor,
                dropout=dropout_actor, hidden_layer_sizes=hidden_layer_sizes_actor, )

        # Critic (Q-Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, l2_reg=l2_reg_critic,
                learn_rate=lr_critic, bn_momentum=bn_momentum_critic, relu_alpha=relu_alpha_critic,
                hidden_layer_sizes=hidden_layer_sizes_critic, dropout=dropout_critic, )
        self.critic_target = Critic(self.state_size, self.action_size, l2_reg=l2_reg_critic,
                learn_rate=lr_critic, bn_momentum=bn_momentum_critic, relu_alpha=relu_alpha_critic,
                hidden_layer_sizes=hidden_layer_sizes_critic, dropout=dropout_critic, )

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = ou_mu
        self.exploration_theta = ou_theta
        self.exploration_sigma = ou_sigma
        self.noise = OUNoise(self.action_size,
                             self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = replay_buffer_size
        self.batch_size = replay_batch_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = discount_factor  # discount factor
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.dropout_actor = dropout_actor
        self.dropout_critic = dropout_critic
        self.bn_momentum_actor = bn_momentum_actor
        self.bn_momentum_critic = bn_momentum_critic
        self.activation_fn_actor = activation_fn_actor
        self.ou_mu=ou_mu
        self.ou_theta=ou_theta
        self.ou_sigma=ou_sigma
        self.replay_buffer_size = replay_buffer_size
        self.replay_batch_size = replay_batch_size
        self.l2_reg_actor = l2_reg_actor
        self.l2_reg_critic = l2_reg_critic
        self.relu_alpha_actor = relu_alpha_actor
        self.relu_alpha_critic = relu_alpha_critic
        self.hidden_layer_sizes_actor = hidden_layer_sizes_actor
        self.hidden_layer_sizes_critic = hidden_layer_sizes_critic

        self.tau_actor = tau_actor
        self.tau_critic = tau_critic

        # Training history
        self.training_scores = []
        self.test_scores = []
        self.training_history = TrainingHistory(env)
        self.q_a_frames_spec = Q_a_frames_spec(env)

        # Track training steps and episodes
        self.steps = 0
        self.episodes = 0

        self.reset_episode()

    def print_summary(self):
        print("Actor model summary:")
        self.actor_local.model.summary()
        print("Critic model summary:")
        self.critic_local.model.summary()
        print("Hyperparameters:")
        print(str(dict(
            train_during_episode=self.train_during_episode,
            discount_factor=self.gamma,
            tau_actor=self.tau_actor, tau_critic=self.tau_critic,
            lr_actor=self.lr_actor, lr_critic=self.lr_critic,
            bn_momentum_actor=self.bn_momentum_actor,
            bn_momentum_critic=self.bn_momentum_critic,
            ou_mu=self.ou_mu, ou_theta=self.ou_theta, ou_sigma=1,
            activation_fn_actor=self.activation_fn_actor,
            replay_buffer_size=self.replay_buffer_size,
            replay_batch_size=self.replay_batch_size,
            l2_reg_actor=self.l2_reg_actor, l2_reg_critic=self.l2_reg_critic,
            relu_alpha_actor=self.relu_alpha_actor,
            relu_alpha_critic=self.relu_alpha_critic,
            dropout_actor=self.dropout_actor, dropout_critic=self.dropout_critic,
            hidden_layer_sizes_actor=self.hidden_layer_sizes_actor,
            hidden_layer_sizes_critic=self.hidden_layer_sizes_critic, )))

    def preprocess_state(self, state):
        obs_space = self.env.observation_space
        return np.array([
            (state[i]-obs_space.low[i])/(obs_space.high[i]-obs_space.low[i])*2 - 1
            for i in range(len(obs_space.low))])

    def reset_episode(self):
        self.noise.reset()
        state = self.preprocess_state(self.env.reset())
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        next_state = self.preprocess_state(next_state)
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and (self.train_during_episode or done):
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
        self.steps += 1

    def act(self, state=None, eps=0, verbose=False):
        """Returns actions for given state(s) as per current policy."""
        if state is None:
            state = self.last_state
        else:
            state = self.preprocess_state(state)
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        noise_sample = self.noise.sample() * max(0,eps)
        res = list(np.clip(action + noise_sample, self.action_low, self.action_high))
        if verbose:
            print("State: (%6.3f, %6.3f), Eps: %6.3f, Action: %6.3f + %6.3f = %6.3f"%
                  (state[0][0], state[0][1], eps, action, noise_sample, res[0]))
        return res  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]
                          ).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]
                          ).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]
                        ).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(
            self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model, self.tau_critic)
        self.soft_update(self.actor_local.model, self.actor_target.model, self.tau_actor)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights),             "Local and target model parameters must have the same size"

        new_weights = tau * local_weights + (1 - tau) * target_weights
        target_model.set_weights(new_weights)

    def train_n_episodes(self, n_episodes, eps=1, eps_decay=None, action_repeat=1,
                         run_tests=True, gen_q_a_frames_every_n_steps=0, draw_plots=False ):
        if eps_decay is None: eps_decay = 1/n_episodes
        n_training_episodes = len(self.training_scores)
        for i_episode in range(n_training_episodes+1, n_training_episodes+n_episodes+1):
            eps -= eps_decay
            eps = max(eps,0)
            episode_start_step = self.steps
            self.run_episode(train=True, action_repeat=action_repeat, eps=eps,
                             gen_q_a_frames_every_n_steps=gen_q_a_frames_every_n_steps )
            if run_tests is True:
                self.run_episode(train=False, eps=0, action_repeat=action_repeat)
            message = "Episode %i - epsilon: %.2f, memory size: %i, training score: %.2f"                        %(self.episodes, eps, len(self.memory), self.training_history.training_episodes[-1].score)
            if run_tests: message += ", test score: %.2f"%self.training_history.test_episodes[-1].score
            print(message)
            sys.stdout.flush()

    def run_episode(self, action_repeat=1, eps=0, train=False, gen_q_a_frames_every_n_steps=0 ):
        next_state = self.reset_episode()
        if train: episode_history = self.training_history.new_training_episode(self.episodes+1,eps)
        else: episode_history = self.training_history.new_test_episode(self.episodes,eps)
        q_a_frame = None
        while True:
            action = self.act(next_state, eps=eps)
            sum_rewards=0
            # Repeat action `action_repeat` times, summing up rewards
            for i in range(action_repeat):
                next_state, reward, done, info = self.env.step(action)
                sum_rewards += reward
                if done:
                    break
            #sum_rewards = np.log1p(sum_rewards)
            episode_history.append(self.steps, next_state, action, sum_rewards)
            if train:
                self.step(action, sum_rewards, next_state, done)
                if gen_q_a_frames_every_n_steps > 0 and self.steps%gen_q_a_frames_every_n_steps==0:
                    self.training_history.add_q_a_frame(self.get_q_a_frames())
            if done:
                if train:
                    self.episodes += 1
                break

    def get_q_a_frames(self):
        """ TODO: Figure out how to work with added dimensions.
                - Use x_dim, y_dim, and a_dim to know which dimensions of state and action to vary.
                    Maybe fill in the unvaried dimensions of states and actions with agent's current state
                    and anticipated action (according to policy).
        """
        xs = self.q_a_frames_spec.xs
        nx = self.q_a_frames_spec.nx
        ys = self.q_a_frames_spec.ys
        ny = self.q_a_frames_spec.ny
        action_space = self.q_a_frames_spec.action_space
        na = self.q_a_frames_spec.na
        x_dim = self.q_a_frames_spec.x_dim
        y_dim = self.q_a_frames_spec.y_dim
        a_dim = self.q_a_frames_spec.a_dim

        def get_state(x,y):
            s=copy.copy(self.last_state)
            s[x_dim]=x
            s[y_dim]=y
            return s
        raw_states = np.array([[ get_state(x,y) for x in xs ] for y in ys ]).reshape(nx*ny, self.state_size)

        def get_action(action):
            a=self.act() if self.action_size>1 else [0]
            a[a_dim]=action
            return a
        actions = np.array([get_action(a) for a in action_space]*nx*ny)

        preprocessed_states = np.array([ self.preprocess_state(s) for s in raw_states])
        Q = self.critic_local.model.predict_on_batch(
            [np.repeat(preprocessed_states,na,axis=0),actions]).reshape((ny,nx,na))
        Q_max = np.max(Q,axis=2)
        Q_std = np.std(Q,axis=2)
        max_action = np.array([action_space[a] for a in np.argmax(Q,axis=2).flatten()]).reshape((ny,nx))
        actor_policy = np.array([ self.act(s)[0] for s in raw_states]).reshape(ny,nx)
        action_gradients = self.critic_local.get_action_gradients(
            [preprocessed_states,actor_policy.reshape(nx*ny,-1),0])[0].reshape(ny,nx)

        return namedtuple( 'q_a_frames',[
                'step_idx', 'episode_idx', 'Q_max', 'Q_std', 'max_action', 'action_gradients', 'actor_policy'
            ])(self.steps, self.episodes, Q_max, Q_std, max_action, action_gradients, actor_policy)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        return random.sample(self.memory, k=min(self.batch_size, len(self)))

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck noise process."""
    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, sigma=None):
        """Update internal state and return it as a noise sample."""
        if sigma is None:
            sigma = self.sigma
        x = self.state
        dx = self.theta * (self.mu - x) + sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, learn_rate,
                 activation_fn, bn_momentum, relu_alpha, l2_reg, dropout, hidden_layer_sizes):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            action_low (array): Min value of each action dimension.
            action_high (array): Max value of each action dimension.
            learn_rate (float): Learning rate.
            activation_fn (string): Activation function, either 'sigmoid' or 'tanh'.
            bn_momentum (float): Batch Normalization momentum .
            relu_alpha (float): LeakyReLU alpha, allowing small gradient when the unit is not active.
            l2_reg (float): L2 regularization factor for each dense layer.
            dropout (float): Dropout rate
            hidden_layer_sizes (list): List of hidden layer sizes.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learn_rate = learn_rate
        self.activation = activation_fn
        self.bn_momentum = bn_momentum
        self.relu_alpha = relu_alpha
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        states = layers.Input(shape=(self.state_size,), name='states')
        net = states

        # Batch Norm instead of input preprocessing (Since we don't know up front the range of state values.)
        #net = layers.BatchNormalization(momentum=self.bn_momentum)(states)

        # Add a hidden layer for each element of hidden_layer_sizes
        for size in self.hidden_layer_sizes:
            net = layers.Dense(units=size, kernel_regularizer=regularizers.l2(l=self.l2_reg))(net)
            #net = layers.BatchNormalization(momentum=self.bn_momentum)(net)
            if self.relu_alpha>0: net = layers.LeakyReLU(alpha=self.relu_alpha)(net)
            else: net = layers.Activation('relu')(net)

        if self.dropout>0: net = layers.Dropout(.2)(net)

        if self.bn_momentum>0: net = layers.BatchNormalization(momentum=self.bn_momentum)(net)

        if self.activation=='tanh':
            # Add final output layer with tanh activation with [-1, 1] output
            actions = layers.Dense(units=self.action_size, activation=self.activation,
                name='actions')(net)
        elif self.activation=='sigmoid':
            # Add final output layer with sigmoid activation
            raw_actions = layers.Dense(units=self.action_size, activation=self.activation,
                name='raw_actions')(net)
            # Scale [0, 1] output for each action dimension to proper range
            actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                name='actions')(raw_actions)
        else:
            raise "Expected 'activation' to be one of: 'tanh', or 'sigmoid'."

        self.model = models.Model(inputs=states, outputs=actions)
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        optimizer = optimizers.Adam(lr=self.learn_rate)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, learn_rate, bn_momentum,
                 relu_alpha, l2_reg, dropout, hidden_layer_sizes,
                ):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            learn_rate (float): Learning rate.
            bn_momentum (float): Batch Normalization momentum.
            relu_alpha (float): LeakyReLU alpha, allowing small gradient when the unit is not active.
            l2_reg (float): L2 regularization factor for each dense layer.
            dropout (float): Dropout rate
            hidden_layer_sizes (list[list]): List of two lists with hidden layer sizes for state and action pathways.
        """
        self.state_size = state_size
        self.action_size = action_size

        assert len(hidden_layer_sizes)==2             and len(hidden_layer_sizes[0])==len(hidden_layer_sizes[1]),            "Expected Critic's hidden_layer_sizes to be a list of two arrays of equal length."

        # Initialize any other variables here
        self.learn_rate = learn_rate
        self.bn_momentum = bn_momentum
        self.relu_alpha = relu_alpha
        self.l2_reg = l2_reg
        self.dropout = dropout
        self.hidden_layer_sizes = hidden_layer_sizes

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        net_states = states
        net_actions = actions

        # Add hidden layer(s) for state pathway
        for size in self.hidden_layer_sizes[0]:
            net_states = layers.Dense(units=size,
                                      kernel_regularizer=regularizers.l2(l=self.l2_reg))(net_states)
            #net_states = layers.BatchNormalization(momentum=self.bn_momentum)(net_states)
            if self.relu_alpha>0: net_states = layers.LeakyReLU(alpha=self.relu_alpha)(net_states)
            else: net_states = layers.Activation('relu')(net_states)

        # Add hidden layer(s) for action pathway
        for size in self.hidden_layer_sizes[1]:
            net_actions = layers.Dense(units=size,
                                       kernel_regularizer=regularizers.l2(l=self.l2_reg))(net_actions)
            #net_actions = layers.BatchNormalization(momentum=self.bn_momentum)(net_actions)
            if self.relu_alpha>0: net_actions = layers.LeakyReLU(alpha=self.relu_alpha)(net_actions)
            else: net_actions = layers.Activation('relu')(net_actions)

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        if self.relu_alpha>0: net = layers.LeakyReLU(alpha=self.relu_alpha)(net)
        else: net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed
        if self.dropout>0: net = layers.Dropout(self.dropout)(net)

        # Normalize the final activations
        if self.bn_momentum>0: net = layers.BatchNormalization(momentum=self.bn_momentum)(net)

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learn_rate)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)


# In[ ]:


class TrainingHistory:
    """
    Tracks training history, including a snapshot of rasterized Q values and actions
    for use in visualizations.
    """
    def __init__(self, env, nx=16, ny=16, na=11, x_dim=0, y_dim=1, a_dim=0):
        """
        Initialize TrainingHistory object with Q grid shape.
        Params
        ======
        """
        self.training_episodes = []
        self.test_episodes = []
        self.q_a_frames = []
        self.last_step = 0
        self.q_a_frames_spec = Q_a_frames_spec(env, nx=nx, ny=ny, na=na,
                                         x_dim=x_dim, y_dim=y_dim, a_dim=a_dim)

    def __repr__(self):
        return "TrainingHistory ( %i training_episodes, %i test_episodes, %i qa_grids, last_step: %i )"%                (len(self.training_episodes), len(self.test_episodes), len(self.q_a_frames), self.last_step)

    def add_q_a_frame(self, q_a_frame):
        self.q_a_frames.append(q_a_frame)

    def new_training_episode(self, idx, epsilon=None):
        episode = EpisodeHistory(idx, epsilon)
        self.training_episodes.append(episode)
        return episode

    def new_test_episode(self, idx, epsilon=None):
        episode = EpisodeHistory(idx, epsilon)
        self.test_episodes.append(episode)
        return episode

    def get_training_episode_for_step(self, step_idx):
        for ep in self.training_episodes:
            if ep.last_step>=step_idx:
                return ep
    def get_test_episode_for_step(self, step_idx):
        for ep in self.test_episodes:
            if (ep.last_step+1)>=step_idx:
                return ep
    def get_q_a_frames_for_step(self, step_idx):
        for g in self.q_a_frames:
            if g.step_idx>=step_idx:
                return g
        return g
    def append_training_step(self, step, state, action, reward):
        """
        Initialize EpisodeHistory with states, actions, and rewards
        Params
        ======
            episode_idx (int): Episode index
            step (int): Step index
            state (list|array): State, array-like of shape env.observation_space.shape
            action (list|array): Action, array-like of shape env.action_space.shape
            reward (float): Reward, scalar value
        """
        if len(self.training_episodes)==0:
            raise "No training episodes exist yet. "
        self.training_episodes[-1].append(step, state, action, reward)
        self.last_step = step
        return self
    def append_test_step(self, step, state, action, reward):
        """
        Initialize EpisodeHistory with states, actions, and rewards
        Params
        ======
            episode_idx (int): Episode index
            step (int): Step index
            state (list|array): State, array-like of shape env.observation_space.shape
            action (list|array): Action, array-like of shape env.action_space.shape
            reward (float): Reward, scalar value
        """
        if len(self.test_episodes)==0:
            raise "No test episodes exist yet. "
        self.test_episodes[-1].append(step, state, action, reward)
        self.last_step = step
        return self
    
class EpisodeHistory:
    """ Tracks the history for a single episode, including the states, actions, and rewards.
    """
    def __init__(self, episode_idx=None, epsilon=None):
        """
        Initialize EpisodeHistory with states, actions, and rewards
        Params
        ======
            episode_idx (int): Episode index
            epsilon (float): Exploration factor
        """
        self.episode_idx = episode_idx
        self.epsilon = epsilon
        self.steps = []
        self.first_step = None
        self.last_step = None
        self.states = []
        self.actions = []
        self.rewards = []

        self.score = self._get_score()

    def _get_score(self):
        return sum(self.rewards)

    def append(self, step, state, action, reward):
        self.steps.append(step)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if self.first_step is None: self.first_step = step
        self.last_step = step
        self.score += reward

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return "EpisodeHistory ( idx: %i, len: %i, first_step: %i, last_step: %i, epsilon: %.3f, score: %.3f )"%            (self.episode_idx,len(self.steps),self.first_step, self.last_step, self.epsilon, self.score)
    
class Q_a_frames_spec():
    """ 
    Tracks training history, including a snapshot of rasterized Q values and actions
    for use in visualizations. 
    """
    def __init__(self, env, nx=16, ny=16, na=11, x_dim=0, y_dim=1, a_dim=0):
        """
        Initialize Q_a_frame_set object with Q grid shape.
        Params
        ======            
             env (obj): OpenAi Gym environment
             nx (int): Width of Q grid (default: 16)
             ny (int): Height of Q grid (default: 16)
             na (int): Depth of Q grid (default: 11)
             x_dim (int): Observation dimension to use as x-axis (default: 0)
             y_dim (int): Observation dimension to use as y-axis (default: 1)
             a_dim (int): Action dimension to use as x-axis (default: 0)
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.a_dim = a_dim
        
        self.xmin = env.observation_space.low[x_dim]
        self.xmax = env.observation_space.high[x_dim]
        self.xs = np.arange(self.xmin, self.xmax, (self.xmax-self.xmin)/nx)[:nx]
        self.nx = len(self.xs)
        
        self.ymin = env.observation_space.low[y_dim]
        self.ymax = env.observation_space.high[y_dim]
        self.ys = np.arange(self.ymin, self.ymax, (self.ymax-self.ymin)/ny)[:ny]
        self.ny = len(self.ys)
        
        self.amin = env.action_space.low[a_dim]
        self.amax = env.action_space.high[a_dim]
        self.action_space = np.linspace(self.amin,self.amax,na)
        self.na = len(self.action_space)


# In[ ]:


from datetime import datetime
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MaxNLocator
from IPython.display import Image, HTML
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
from imageio_ffmpeg import get_ffmpeg_exe
import numpy as np

# plt.rcParams['animation.embed_limit'] = 200
plt.rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()

plt.rcParams['axes.labelsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 10

def create_animation(agent, every_n_steps=1, display_mode='gif', fps=30):
    history = agent.training_history
    fig = plt.figure(figsize=(11,6))
    fig.set_tight_layout(True)
    main_rows = gridspec.GridSpec(2, 1, figure=fig, top=.9, left=.05, right=.95, bottom=.25)

    def create_top_row_im(i, title='', actions_cmap=False):
        top_row = main_rows[0].subgridspec(1, 5, wspace=.3)
        ax = fig.add_subplot(top_row[i])
        ax.axis('off')
        ax.set_title(title)
        im = ax.imshow( np.zeros((len(history.q_a_frames_spec.ys),
                                  len(history.q_a_frames_spec.xs))), origin='lower' )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if actions_cmap is True:
            im.set_clim(history.q_a_frames_spec.amin,
                        history.q_a_frames_spec.amax)
            im.set_cmap("RdYlGn")
            cb = fig.colorbar(im, cax=cax)
        else:
            cb = fig.colorbar(im, cax=cax, format='%.3g')
        cb.ax.tick_params(labelsize=8)
        return im

    def create_bottom_row_plot(i, title=''):
        bottom_row = main_rows[1].subgridspec(1,3)
        ax = fig.add_subplot(bottom_row[i])
        ax.set_title(title)
        return ax

    Q_max_im = create_top_row_im(0, title='Q max')
    Q_std_im = create_top_row_im(1, title='Q standard deviation')
    action_gradients_im = create_top_row_im(2, title="Action Gradients")
    max_action_im = create_top_row_im(3, title="Action with Q max", actions_cmap=True)
    actor_policy_im = create_top_row_im(4, title="Policy", actions_cmap=True)

    scores_ax = create_bottom_row_plot(0, title="Scores")
    scores_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    scores_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    training_scores_line, = scores_ax.plot([], 'bo', label='training')
    test_scores_line, = scores_ax.plot([], 'ro', label='test')
    scores_ax.set_xlim(1,len(history.training_episodes))
    scores_combined = np.array([e.score for e in history.training_episodes ]+                               [e.score for e in history.test_episodes ])
    scores_ax.set_ylim(scores_combined.min(),scores_combined.max())
    scores_ax.set_xlabel('episode')
    scores_ax.set_ylabel('total reward')
    scores_ax.legend(loc='upper left', bbox_to_anchor=(0,-.1))

    training_episode_ax = create_bottom_row_plot(1)
    training_episode_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    training_episode_position_line, = training_episode_ax.plot([], 'b-', label='position')
    training_episode_velocity_line, = training_episode_ax.plot([], 'm-', label='velocity')
    training_episode_action_line, = training_episode_ax.plot([], 'g-', label='action')
    training_episode_reward_line, = training_episode_ax.plot([], 'r-', label='reward')
    training_episode_ax.set_ylim((-1.1,1.1))
    training_episode_ax.axes.get_yaxis().set_visible(False)
#     training_episode_ax.legend(loc='upper left', ncol=2)

    test_episode_ax = create_bottom_row_plot(2)
    test_episode_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    test_episode_position_line, = test_episode_ax.plot([], 'b-', label='position')
    test_episode_velocity_line, = test_episode_ax.plot([], 'm-', label='velocity')
    test_episode_action_line, = test_episode_ax.plot([], 'g-', label='action')
    test_episode_reward_line, = test_episode_ax.plot([], 'r-', label='reward')
    test_episode_ax.set_ylim((-1.1,1.1))
    test_episode_ax.axes.get_yaxis().set_visible(False)
    test_episode_ax.legend(loc='upper left', ncol=2, bbox_to_anchor=(-.5,-.1))


    def update(step_idx):
        num_frames = math.ceil(last_step/every_n_steps)
        frame_idx = math.ceil(step_idx/every_n_steps)
        print("Drawing frame: %i/%i, %.2f%%\r"%              (frame_idx+1, num_frames, 100*(frame_idx+1)/float(num_frames) ), end='')
        training_episode = history.get_training_episode_for_step(step_idx)
        episode_step_idx = step_idx - training_episode.first_step

        q_a_frames = history.get_q_a_frames_for_step(step_idx)

        Q_max_im.set_data(q_a_frames.Q_max)
        Q_max_im.set_clim(q_a_frames.Q_max.min(),q_a_frames.Q_max.max())
        Q_std_im.set_data(q_a_frames.Q_std)
        Q_std_im.set_clim(q_a_frames.Q_std.min(),q_a_frames.Q_std.max())
        action_gradients_im.set_data(q_a_frames.action_gradients)
        action_gradients_im.set_clim(q_a_frames.action_gradients.min(),
                                     q_a_frames.action_gradients.max())
        max_action_im.set_data(q_a_frames.max_action)
        actor_policy_im.set_data(q_a_frames.actor_policy)

        # Plot scores
        xdata = range(1,training_episode.episode_idx+1)
        training_scores_line.set_data(xdata,
            [e.score for e in history.training_episodes ][:training_episode.episode_idx] )
        test_scores_line.set_data(xdata,
            [e.score for e in history.test_episodes ][:training_episode.episode_idx] )

        #Plot training episode
        training_episode_ax.set_title("Training episode %i, eps=%.3f, score: %.3f"%(
                            training_episode.episode_idx, training_episode.epsilon, training_episode.score))

        current_end_idx = episode_step_idx + every_n_steps
        if current_end_idx >= len(training_episode.states):
            current_end_idx = len(training_episode.states)-1

        training_xdata = range(0,current_end_idx+1)
        training_episode_ax.set_xlim(training_xdata[0],
                                     training_episode.last_step-training_episode.first_step+1)
        episode_states = [agent.preprocess_state(s) for s in training_episode.states]
        training_episode_position_line.set_data(training_xdata,
                                                [s[0] for s in episode_states][:current_end_idx+1])
        training_episode_velocity_line.set_data(training_xdata,
                                                [s[1] for s in episode_states][:current_end_idx+1])
        training_episode_action_line.set_data(training_xdata,
                                              training_episode.actions[:current_end_idx+1])
        training_episode_reward_line.set_data(training_xdata,
                                              training_episode.rewards[:current_end_idx+1])

        #Plot test episode
        test_episode = history.get_test_episode_for_step(step_idx)
        if test_episode is not None:
            test_episode_ax.set_title("Test episode %i, score: %.3f"%(
                                test_episode.episode_idx, test_episode.score))
            test_xdata = range(1,len(test_episode.states)+1)
            test_episode_ax.set_xlim(test_xdata[0],test_xdata[-1])
            episode_states = [agent.preprocess_state(e) for e in test_episode.states]
            test_episode_position_line.set_data(test_xdata, [s[0] for s in episode_states])
            test_episode_velocity_line.set_data(test_xdata, [s[1] for s in episode_states])
            test_episode_action_line.set_data(test_xdata,
                                              test_episode.actions )
            test_episode_reward_line.set_data(test_xdata, test_episode.rewards )

    last_step = history.training_episodes[-1].last_step + 1
    anim = FuncAnimation(fig, update, interval=1000/fps,
                         frames=range(0,last_step,every_n_steps))

    if display_mode=='video' or display_mode=='video_file':
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
        if writer.isAvailable():
            print("Using ffmpeg at '%s'."%writer.bin_path())
        else:
            raise("FFMpegWriter not available for video output.")
    if display_mode=='js':
        display(HTML(anim.to_jshtml()))
    elif display_mode=='video':
        display(HTML(anim.to_html5_video()))
    elif display_mode=='video_file':
        filename = 'training_animation_%i.mp4'%int(datetime.now().timestamp())
        img = anim.save(filename, writer=writer)
        print("\rVideo saved to %s."%filename)
        import io, base64
        encoded = base64.b64encode(io.open(filename, 'r+b').read())
        display(HTML(data='''<video alt="training animation" controls loop autoplay>
                        <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                     </video>'''.format(encoded.decode('ascii'))))
    else:
        filename = 'training_animation_%i.gif'%int(datetime.now().timestamp())
        img = anim.save(filename, dpi=80, writer='imagemagick')
        display(HTML("<img src='%s'/>"%filename))
    plt.close()


# In[ ]:


agent = DDPG(env, train_during_episode=True, ou_mu=0, ou_theta=.05, ou_sigma=.25, 
             discount_factor=.999, replay_buffer_size=10000, replay_batch_size=1024,
             tau_actor=.3, tau_critic=.1, 
             relu_alpha_actor=.01, relu_alpha_critic=.01,
             lr_actor=.0001, lr_critic=.005, activation_fn_actor='tanh',
             l2_reg_actor=.01, l2_reg_critic=.01, 
             bn_momentum_actor=0, bn_momentum_critic=.7,
             hidden_layer_sizes_actor=[16,32,16], hidden_layer_sizes_critic=[[16,32],[16,32]], )
agent.print_summary()


# In[ ]:


agent.train_n_episodes(50, eps=1, eps_decay=1/50, action_repeat=5, 
                       run_tests=True, gen_q_a_frames_every_n_steps=10, )


# In[ ]:


create_animation(agent, display_mode='video_file', every_n_steps=10, fps=15)

