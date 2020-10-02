# %% [code]
import numpy as np
import matplotlib.pylab as plt


class POMDPModel(object):
    def __init__(self, states, actions, observations, starts=None, terminal=None):
        self.states = states
        self.actions = actions
        self.observations = observations

        self.n_states = len(states)
        self.n_actions = len(actions)
        self.n_observations = len(observations)

        self.starts = starts
        self.terminal = terminal

        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.O = np.zeros((self.n_states, self.n_actions, self.n_observations))
        self.R = np.zeros((self.n_states, self.n_actions))

    def check_model(self):
        for s in range(self.n_states):
            if self.terminal is None or s not in self.terminal:
                for a in range(self.n_actions):
                    if abs(sum(self.T[s, a, :]) - 1) >= 1e-10:
                        raise Exception('The transition probability for state %d and action %s should sum as 1' % (
                            s, self.actions[a]))

                    if abs(sum(self.O[s, a, :]) - 1) >= 1e-10:
                        raise Exception('The observation probability for state %d and action %s should sum as 1' % (
                            s, self.actions[a]))

        return True

    def update_belief(self, belief, a, o):
        bp = np.zeros(self.n_states)

        if isinstance(a, str):
            a = self.actions.index(a)

        if isinstance(o, str):
            o = self.observations.index(o)

        for spi in range(self.n_states):
            v = np.sum(self.T[:, a, spi] * belief)
            bp[spi] = self.O[spi, a, o] * v

        return bp / sum(bp)

    def show_alpha_vector(self, gamma):
        for av in gamma:
            print(self.actions[av.action], av.value)

    def plot_alpha_vector(self, gamma):
        if self.n_states > 2:
            print('Only can plot two-dimensional alpha vectors')
            return

        for av in gamma:
            plt.plot([0, 1], av.value, label=self.actions[av.action])

        plt.xlabel('P(%s)' % self.states[1])
        plt.legend()


class BabyPOMDP(POMDPModel):
    def __init__(self):
        states = ['not-hungry', 'hungry']
        actions = ['not-feed', 'feed']
        observations = ['not-crying', 'crying']

        super().__init__(states, actions, observations)

        self.T = np.array([[[0.9, 0.1], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]])
        self.R = np.array([[0.0, -5], [-10, -15]])
        self.O = np.array([[[0.9, 0.1], [0.9, 0.1]], [[0.2, 0.8], [0.2, 0.8]]])


class TigerPOMDP(POMDPModel):
    def __init__(self):
        states = ['left', 'right']
        actions = ['listening', 'opening-left', 'opening-right']
        observations = ['left', 'right']

        super().__init__(states, actions, observations)

        self.T = np.array([[[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]], [
                          [0.0, 1.0], [0.5, 0.5], [0.5, 0.5]]])
        self.R = np.array([[-1, -100, 10], [-1, 10, -100]])
        self.O = np.array([[[0.85, 0.15], [0.5, 0.5], [0.5, 0.5]], [
                          [0.15, 0.85], [0.5, 0.5], [0.5, 0.5]]])
