# %% [code]
import numpy as np
from itertools import product
from scipy.optimize import linprog


class AlphaVector:
    def __init__(self, a, v):
        self.action = a
        self.value = v

    def copy(self):
        return AlphaVector(self.action, self.value)


class POMDPAgent(object):
    def __init__(self, discount=0.9, threshold=0.0001, max_iter=10000, prune_eps=-1e-14, verbose=False):
        self.discount = discount
        self.threshold = threshold
        self.max_iter = max_iter
        self.prune_eps = prune_eps
        self.verbose = verbose

        self.gamma = set()

    def reset(self):
        self.gamma = set()

    def choose_action(self, belief):
        max_v = -np.inf
        best = None
        for av in self.gamma:
            v = np.dot(av.value, belief)

            if v > max_v:
                max_v = v
                best = av

        if best is None:
            raise ValueError('Vector set should not be empty')

        return best.action, best

    def value_iteration(self, model, horizon, prune=True):
        dummy = AlphaVector(a=-1, v=np.zeros(model.n_states))
        self.gamma.add(dummy)

        for h in range(horizon):
            gamma_h = set()

            v_new = np.zeros(
                shape=(len(self.gamma), model.n_actions, model.n_observations, model.n_states))
            idx = 0

            for av in self.gamma:
                for ai in range(model.n_actions):
                    for oi in range(model.n_observations):
                        for si in range(model.n_states):
                            v_new[idx, ai, oi, si] = np.sum(
                                av.value * model.O[:, ai, oi] * model.T[si, ai, :])
                idx += 1

            for ai in range(model.n_actions):
                for indicies in [p for p in product(list(range(idx)), repeat=model.n_observations)]:
                    temp = np.zeros(model.n_states)
                    for si in range(model.n_states):
                        temp[si] = model.R[si, ai] + self.discount * np.sum(
                            [v_new[indicies[oi], ai, oi, si] for oi in range(model.n_observations)])

                    gamma_h.add(AlphaVector(a=ai, v=temp))

            self.gamma = gamma_h.copy()
            if prune == True:
                self.prune(model.n_states)

            print('Level %d # of alpha vectors: %d' % (h+1, len(self.gamma)))
            if self.verbose:
                for av in self.gamma:
                    print(model.actions[av.action], ': ', av.value)

    def prune(self, n_states):
        F = self.gamma.copy()
        Q = set()

        for i in range(n_states):
            max_i = -np.inf
            best = None

            for av in F:
                if av.value[i] > max_i:
                    max_i = av.value[i]
                    best = av

            if best is not None:
                Q.update({best})
                F.remove(best)

        while F:
            av_i = F.pop()
            F.add(av_i)

            dominated = False

            c = np.append(-av_i.value, 1)

            A_eq = np.append(np.ones(n_states), 0).reshape(1, n_states+1)
            b_eq = np.array([1])

            bounds = [(0, None) for _ in range(n_states)]
            bounds.append((None, None))

            A_ub = np.zeros((len(Q), n_states+1))
            b_ub = np.zeros(len(Q))

            for j, av_j in enumerate(Q):
                A_ub[j, :-1] = av_j.value
                A_ub[j, -1] = -1

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                          b_eq=b_eq, bounds=bounds, method='revised simplex')

            if res.fun >= self.prune_eps:
                dominated = True
                F.remove(av_i)

            if not dominated:
                max_k = -np.inf
                best = None
                for av_k in F:
                    b = res['x'][0:n_states]
                    v = np.dot(av_k.value, b)
                    if v > max_k:
                        max_k = v
                        best = av_k
                F.remove(best)
                Q.update({best})

        self.gamma = Q.copy()

    def qmdp(self, model):
        for ai in range(model.n_actions):
            self.gamma.add(AlphaVector(a=ai, v=np.zeros(model.n_states)))

        for h in range(self.max_iter):
            gamma_h = set()

            for ai in range(model.n_actions):
                value = np.zeros(model.n_states)

                for si in range(model.n_states):
                    max_v = np.array(
                        [np.max([av.value[spi] for av in self.gamma]) for spi in range(model.n_states)])
                    value[si] = model.R[si][ai] + self.discount * \
                        np.sum(model.T[si, ai, :] * max_v)

                gamma_h.add(AlphaVector(a=ai, v=value))

            delta = self.value_difference(gamma_h)

            self.gamma = gamma_h.copy()
            print('Iteration %d value difference %f' % (h, delta))
            if self.verbose:
                for av in self.gamma:
                    print(av.action, av.value)

            if delta <= self.threshold:
                break

    def fib(self, model):
        for ai in range(model.n_actions):
            self.gamma.add(AlphaVector(a=ai, v=np.zeros(model.n_states)))

        for h in range(self.max_iter):
            gamma_h = set()

            for ai in range(model.n_actions):
                value = np.zeros(model.n_states)

                for si in range(model.n_states):
                    value[si] = model.R[si, ai]

                    for oi in range(model.n_observations):
                        max_v = np.max(
                            [np.sum(model.O[:, ai, oi] * model.T[si, ai, :] * av.value) for av in self.gamma])

                        value[si] += self.discount * max_v

                gamma_h.add(AlphaVector(a=ai, v=value))

            delta = self.value_difference(gamma_h)

            self.gamma = gamma_h.copy()
            print('Iteration %d value difference %f' % (h, delta))
            if self.verbose:
                for av in self.gamma:
                    print(av.action, av.value)

            if delta <= self.threshold:
                break

    def backup_belief(self, model, belief):
        max_v = -np.inf
        best = None

        for ai in range(model.n_actions):
            alpha_ao = []
            for oi in range(model.n_observations):
                bao = model.update_belief(belief, ai, oi)
                alpha_ao.append(self.choose_action(bao)[1])

            v = np.zeros(model.n_states)
            for si in range(model.n_states):
                sp_sum = 0
                for spi in range(model.n_states):
                    sp_sum += model.T[si, ai, spi] * np.sum(
                        [model.O[spi, ai, oi] * alpha_ao[oi].value[spi] for oi in range(model.n_observations)])

                v[si] = model.R[si, ai] + self.discount * sp_sum

            if np.sum(v * belief) > max_v:
                max_v = np.sum(v * belief)
                best = AlphaVector(a=ai, v=v)

        return best

    def value_difference(self, gamma_h):
        delta = 0.

        gamma_dict = dict()

        for av in self.gamma:
            gamma_dict.update({av.action: av.value})

        for av in gamma_h:
            delta += np.sum(np.abs(av.value-gamma_dict[av.action]))

        return delta
