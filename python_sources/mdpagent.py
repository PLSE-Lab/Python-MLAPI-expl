# %% [code]
import numpy as np
from tqdm import tqdm
import random
from math import floor, ceil


class MDP(object):
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

    def reset(self):
        self.T = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.R = np.zeros((self.n_states, self.n_actions))

    def update_parameter(self, N, Nsa, rho, s, a):
        self.T[s, a, :] = N[s, a, :] / Nsa[s, a]
        self.R[s, a] = rho[s, a] / Nsa[s, a]


class Agent(object):
    def __init__(self, n_states, n_actions, action_sets=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.action_sets = action_sets

        self.reset()

    def reset_infeasible_actions(self, s=None):
        if self.action_sets is not None:
            if s is not None:
                if s in self.action_sets:
                    self.Q[s, self.action_sets[s][1]] = -1e100
            else:
                for s, a in self.action_sets.items():
                    self.Q[s, a[1]] = -1e100

    def reset(self):
        self.U = np.zeros(self.n_states)
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.pi = np.zeros(self.n_states)

        self.reset_infeasible_actions()

    def check_policy(self, pi):
        if self.action_sets is not None:
            for s, a in self.action_sets.items():
                if len(pi.shape) == 1:
                    if pi[s] in a[1]:
                        raise Exception('Infeasible policy')
                else:
                    if sum(pi[s, a[1]]) != 0:
                        raise Exception('Infeasible policy')

    def random_policy(self):
        pi = np.random.choice(range(self.n_actions), size=self.n_states)
        if self.action_sets is not None:
            for s, a in self.action_sets.items():
                pi[s] = np.random.choice(a[0])

        return pi

    def stochastic_policy(self, pi):
        self.check_policy(pi)
        return np.array([[1 if pi[s] == a else 0 for a in range(self.n_actions)] for s in range(self.n_states)])

    def extract_policy(self):
        self.pi = np.argmax(self.Q, axis=1)
        self.check_policy(self.pi)

    def state_value(self, pi=None):
        if pi is None:
            self.U = np.max(self.Q, axis=1)
        else:
            self.check_policy(pi)

            if len(pi.shape) == 1:
                pi = self.stochastic_policy(pi)

            self.U = np.sum(pi * self.Q, axis=1)


class MDPAgent(Agent):
    def __init__(self, n_states, n_actions, action_sets=None, discount=0.9, threshold=0.0001, max_iter=1000, verbose=False):
        super().__init__(n_states, n_actions, action_sets)

        self.discount = discount
        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose

    def action_value(self, model):
        self.Q = model.R + self.discount * model.T.dot(self.U)
        self.reset_infeasible_actions()

    def iterative_policy_evaluation(self, model, pi):
        self.check_policy(pi)

        if len(pi.shape) == 1:
            pi = self.stochastic_policy(pi)

        for iter in range(1, self.max_iter+1):
            prevU = self.U.copy()

            self.action_value(model)
            self.state_value(pi)

            delta = np.sum(np.abs(self.U - prevU))

            if self.verbose:
                print('Iterative Policy Evaluation: iteration %d, delta %f' %
                      (iter, delta))

            if delta <= self.threshold:
                break

    def policy_iteration(self, model, pi=None):
        if pi is None:
            pi = self.random_policy()
        else:
            self.check_policy(pi)

        for iter in range(1, self.max_iter+1):
            prevPi = pi.copy()

            self.iterative_policy_evaluation(model, pi)
            self.extract_policy()
            pi = self.pi.copy()

            delta = np.sum(np.abs(pi - prevPi))

            if self.verbose:
                print('Policy Iteration: iteration %d, delta %f' %
                      (iter, delta))

            if delta == 0:
                break

    def value_iteration(self, model):
        for iter in range(1, self.max_iter+1):
            prevU = self.U.copy()

            self.action_value(model)
            self.state_value()

            delta = np.sum(np.abs(self.U - prevU))

            if self.verbose:
                print('Value Iteration: iteration %d, delta %f' % (iter, delta))

            if delta <= self.threshold:
                break

    def value_iteration_gs(self, model):
        for iter in range(1, self.max_iter+1):
            prevU = self.U.copy()

            for s in range(self.n_states):
                self.Q[s, :] = model.R[s, :] + \
                    self.discount * model.T[s, :, :].dot(self.U)
                self.reset_infeasible_actions(s)
                self.U[s] = np.max(self.Q[s, :])

            delta = np.sum(np.abs(self.U - prevU))

            if self.verbose:
                print('Gauss-Siedel Value Iteration: iteration %d, delta %f' %
                      (iter, delta))

            if delta <= self.threshold:
                break


class RLAgent(Agent):
    def __init__(self, n_states, n_actions, action_sets=None, discount=0.9, epsilon=0.2, verbose=False):
        super().__init__(n_states, n_actions, action_sets)

        self.discount = discount
        self.epsilon = epsilon
        self.verbose = verbose

    def choose_action(self, env, s, pi=None):
        if pi is None:
            if random.random() <= self.epsilon:
                action_sets = range(self.n_actions)
                if env.action_sets is not None and s in env.action_sets:
                    action_sets = env.action_sets[s][0]
                return random.choices(action_sets, k=1)[0]
            else:
                return np.argmax(self.Q[s, :])
        else:
            self.check_policy(pi)

            if len(pi.shape) == 1:
                pi = self.stochastic_policy(pi)

            return random.choices(range(self.n_actions), weights=pi[s, :], k=1)[0]

    def generate_episode(self, env, n_steps, pi=None):
        if pi is not None:
            self.check_policy(pi)

        trajectory = []

        s = env.initialize_state()

        for _ in range(n_steps):
            a = self.choose_action(env, s, pi)
            next_s, r, done = env.step(s, a)

            trajectory.append((s, a, r, next_s, done))

            s = next_s

            if done:
                break

        return trajectory


class ModelRLAgent(MDPAgent, RLAgent):
    def __init__(self, n_states, n_actions, action_sets=None, discount=0.9, threshold=0.0001, max_iter=1000, epsilon=0.2, verbose=False):
        MDPAgent.__init__(self, n_states, n_actions, action_sets,
                          discount, threshold, max_iter, verbose)
        RLAgent.__init__(self, n_states, n_actions,
                         action_sets, discount, epsilon, verbose)

        self.N = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.Nsa = np.zeros((self.n_states, self.n_actions))
        self.rho = np.zeros((self.n_states, self.n_actions))

    def learning(self, model, s, a, r, next_s):
        self.N[s, a, next_s] += 1
        self.Nsa[s, a] += 1
        self.rho[s, a] += r

        model.update_parameter(self.N, self.Nsa, self.rho, s, a)
        self.value_iteration(model)

    def run_episodes(self, env, model, n_episodes, n_steps, disable_bar=False):
        for _ in tqdm(range(n_episodes), disable=disable_bar):
            s = env.initialize_state()

            for _ in range(n_steps):
                a = self.choose_action(env, s)
                next_s, r, done = env.step(s, a)

                self.learning(model, s, a, r, next_s)

                s = next_s

                if done:
                    break


class MonteCarloRLAgent(RLAgent):
    def __init__(self, n_states, n_actions, action_sets=None, discount=0.9, epsilon=0.2, alpha=0.1, verbose=False):
        super().__init__(n_states, n_actions, action_sets, discount, epsilon, verbose)

        self.alpha = alpha

    def action_value(self, returns, counts):
        self.Q = np.divide(returns, counts, out=np.zeros_like(
            returns), where=counts != 0)
        self.reset_infeasible_actions()

    def learning(self, n_episodes, n_steps, env, pi, method='every-visit', purpose='control', disable_bar=False):
        self.check_policy(pi)

        returns = np.zeros((self.n_states, self.n_actions), dtype=float)
        counts = np.zeros((self.n_states, self.n_actions), dtype=int)
        values = []

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            trajectory = self.generate_episode(env, n_steps, pi)

            v = 0
            for i, (s, a, r, _, _) in enumerate(trajectory[::-1]):
                v = self.discount * v + r
                if method == 'first-visit':
                    history = [(sp, ap)
                               for sp, ap, _, _, _ in trajectory[:-1-i]]
                    if (s, a) not in history:
                        returns[s, a] += v
                        counts[s, a] += 1
                else:
                    returns[s, a] += v
                    counts[s, a] += 1

            self.action_value(returns, counts)

            if purpose == 'control':
                pi = self.epsilon_policy()
                self.state_value()
            elif purpose == 'prediction':
                self.state_value(pi=pi)

            if self.verbose == True:
                values.append(self.U.copy())

        return values

    def td_learning(self, n_episodes, n_steps, env, pi, disable_bar=False):
        self.check_policy(pi)

        values = []

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            trajectory = self.generate_episode(env, n_steps, pi)

            for i, (s, _, _, _, _) in enumerate(trajectory):
                self.U[s] += self.alpha * (sum([r * pow(self.discount, j)
                                                for j, (_, _, r, _, _) in enumerate(trajectory[i:])]) - self.U[s])

            if self.verbose == True:
                values.append(self.U.copy())

        return values

    def epsilon_policy(self):
        pi = np.ones((self.n_states, self.n_actions)) * \
            self.epsilon / self.n_actions
        for s, a in enumerate(np.argmax(self.Q, axis=1)):
            pi[s, a] = 1 - self.epsilon + self.epsilon / self.n_actions

        if self.action_sets is not None:
            for s, a in self.action_sets.items():
                pi[s, a[1]] = 0
                pi[s, a[0]] = self.epsilon / len(a[0])
                pi[s, np.argmax(self.Q[s, :])] = 1 - \
                    self.epsilon + self.epsilon / len(a[0])

        return pi

    def iterative_policy_evaluation(self, n_episodes, n_steps, env, pi, method='every-visit'):
        return self.learning(n_episodes, n_steps, env, pi, method=method, purpose='prediction')

    def policy_iteration(self, n_episodes, n_steps, env, pi=None, method='every-visit'):
        if pi is None:
            pi = self.epsilon_policy()

        return self.learning(n_episodes, n_steps, env, pi, method=method, purpose='control')


class TDLearningAgent(RLAgent):
    def __init__(self, n_states, n_actions, action_sets=None, discount=0.9, epsilon=0.2, alpha=0.1, verbose=False):
        super().__init__(n_states, n_actions, action_sets, discount, epsilon, verbose)

        self.alpha = alpha

    def iterative_policy_evaluation(self, n_episodes, n_steps, env, pi, disable_bar=False):
        self.check_policy(pi)

        values = []

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            s = env.initialize_state()

            for _ in range(n_steps):
                a = self.choose_action(env, s, pi)
                next_s, r, done = env.step(s, a)

                self.U[s] += self.alpha * \
                    (r + self.discount * self.U[next_s] - self.U[s])

                s = next_s

                if done:
                    break

            if self.verbose:
                values.append(self.U.copy())

        return values

    def sarsa(self, n_episodes, n_steps, env, disable_bar=False):
        rewards = []

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            s = env.initialize_state()
            a = self.choose_action(env, s)

            reward = 0

            for _ in range(n_steps):
                next_s, r, done = env.step(s, a)
                next_a = self.choose_action(env, next_s)

                reward += r

                self.Q[s, a] += self.alpha * \
                    (r + self.discount * self.Q[next_s, next_a] - self.Q[s, a])

                s = next_s
                a = next_a

                if done:
                    break

            rewards.append(reward)

        return rewards

    def q_learning(self, n_episodes, n_steps, env, disable_bar=False):
        rewards = []

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            s = env.initialize_state()

            reward = 0

            for _ in range(n_steps):
                a = self.choose_action(env, s)
                next_s, r, done = env.step(s, a)

                reward += r

                self.Q[s, a] += self.alpha * \
                    (r + self.discount *
                     np.max(self.Q[next_s, :]) - self.Q[s, a])

                s = next_s

                if done:
                    break

            rewards.append(reward)

        return rewards


class ApproximateRLAgent(RLAgent):
    def __init__(self, n_states, n_actions, approximate_func, action_sets=None, discount=0.9, epsilon=0.2, alpha=0.1, beta=0.1, verbose=False):
        super().__init__(n_states, n_actions, action_sets, discount, epsilon, verbose)

        self.alpha = alpha
        self.beta = beta
        self.approximate_func = approximate_func

        self.average_reward = 0

    def approximate_state_value(self, env):
        self.U = np.array([self.approximate_func.state_value(env, s)
                           for s in range(env.n_states)])

    def approximate_action_value(self, env, s=None):
        if s is None:
            self.Q = np.array([[self.approximate_func.action_value(env, s, a) for a in range(env.n_actions)]
                               for s in range(env.n_states)])
        else:
            self.Q[s, :] = np.array([self.approximate_func.action_value(
                env, s, a) for a in range(env.n_actions)])

    def gradient_monte_carlo_policy_evaluation(self, n_episodes, n_steps, env, pi, disable_bar=False):
        self.check_policy(pi)

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            trajectory = self.generate_episode(env, n_steps, pi)

            for i, (s, _, _, _, _) in enumerate(trajectory):
                delta = self.alpha * (sum([r * pow(self.discount, j) for j, (_, _, r, _, _) in enumerate(
                    trajectory[i:])]) - self.approximate_func.state_value(env, s))
                self.approximate_func.update_parameters(delta, env, s)

    def semigradient_temporal_difference_policy_evaluation(self, n_episodes, n_steps, env, pi, disable_bar=False):
        self.check_policy(pi)

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            s = env.initialize_state()

            for _ in range(n_steps):
                a = self.choose_action(env, s, pi)
                next_s, r, done = env.step(s, a)

                delta = self.alpha * (r + self.discount * self.approximate_func.state_value(
                    env, next_s) - self.approximate_func.state_value(env, s))

                self.approximate_func.update_parameters(delta, env, s)

                s = next_s

                if done:
                    break

    def differential_semigradient_sarsa(self, n_steps, env, disable_bar=False):
        s = env.initialize_state()
        a = self.choose_action(env, s)

        for _ in tqdm(range(n_steps), disable=disable_bar):
            next_s, r, done = env.step(s, a)

            self.approximate_action_value(env, next_s)
            next_a = self.choose_action(env, next_s)

            delta = r - self.average_reward + self.approximate_func.action_value(
                env, next_s, next_a) - self.approximate_func.action_value(env, s, a)
            self.average_reward += self.beta * delta

            self.approximate_func.update_parameters(
                self.alpha*delta, env, s, a)

            s = next_s
            a = next_a

            if done:
                break


class ValueFunction(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def state_value(self, env, s):
        pass

    def action_value(self, env, s, a):
        pass

    def update_parameters(self, delta, env, s, a=None):
        pass


class AggregateValueFunction(ValueFunction):
    def __init__(self, n_groups, n_states):
        self.n_groups = n_groups
        self.group_size = ceil(n_states / n_groups)
        self.parameters = np.zeros(n_groups)

    def reset(self):
        self.parameters = np.zeros(self.n_groups)

    def state_value(self, env, s):
        if env.terminal is not None and s in env.terminal:
            return 0

        if env.walls is not None and s in env.walls:
            return 0

        group_index = s // self.group_size
        return self.parameters[group_index]

    def update_parameters(self, delta, env, s, a=None):
        group_index = s // self.group_size
        self.parameters[group_index] += delta


class IHT(object):
    def __init__(self, size_val):
        self.size = size_val
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull_count == 0:
                print('IHT full, starting to allow collisions')
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


class TileCodingValueFunction(ValueFunction):
    def __init__(self, n_tilings, max_size):
        self.n_tilings = n_tilings
        self.max_size = max_size
        self.hash_table = IHT(self.max_size)
        self.parameters = np.zeros(self.max_size)

    def reset(self):
        self.hash_table = IHT(self.max_size)
        self.parameters = np.zeros(self.max_size)

    def get_active_tiles(self, floats, ints=None, read_only=False):
        if ints is None:
            ints = []
        else:
            ints = [ints]
        qfloats = [floor(f * self.n_tilings * self.n_tilings) for f in floats]
        tiles = []
        for tiling in range(self.n_tilings):
            tilingX2 = tiling * 2
            coords = [tiling]
            b = tiling
            for q in qfloats:
                coords.append((q + b) // self.n_tilings)
                b += tilingX2
            coords.extend(ints)

            tiles.append(self.hash_table.get_index(tuple(coords), read_only))
        return tiles

    def state_value(self, env, s):
        values = [self.action_value(env, s, a) for a in range(env.n_actions)]
        return np.max(values)

    def action_value(self, env, s, a):
        if env.terminal is not None and s in env.terminal:
            return 0

        if env.walls is not None and s in env.walls:
            return 0

        if env.action_sets is not None and s in env.action_sets:
            if a in env.action_sets[s][1]:
                return -1e100

        active_tiles = self.get_active_tiles(env.s2tile(s), a)
        return np.sum(self.parameters[active_tiles])

    def update_parameters(self, delta, env, s, a=None):
        active_tiles = self.get_active_tiles(env.s2tile(s), a)
        for active_tile in active_tiles:
            self.parameters[active_tile] += delta / self.n_tilings


class PolicyGradientAgent(RLAgent):
    def __init__(self, n_states, n_actions, policy, action_sets=None, discount=0.9, epsilon=0.2, alpha=0.1, verbose=False):
        super().__init__(n_states, n_actions, action_sets, discount, epsilon, verbose)

        self.alpha = alpha
        self.policy = policy

    def reinforce(self, n_episodes, n_steps, env, disable_bar=False):
        rewards = []

        for _ in tqdm(range(n_episodes), disable=disable_bar):
            rewards_sum = 0

            pi = self.policy.get_pi(env, self.epsilon)

            trajectory = self.generate_episode(env, n_steps, pi)

            for i, (_, a, r, _, _) in enumerate(trajectory):
                rewards_sum += r
                delta = self.alpha * pow(self.discount, i) * sum([rp * pow(self.discount, j)
                                                                  for j, (_, _, rp, _, _) in enumerate(trajectory[i:])])
                self.policy.update_parameters(env, self.epsilon, delta, a)

            if self.verbose == True:
                rewards.append(rewards_sum)

        return rewards


class Policy(object):
    def __init__(self, theta, beta):
        self.theta = theta.copy()
        self.beta = beta.copy()

    def reset(self, theta):
        self.theta = theta

    def get_pmf(self, env, epsilon):
        h = np.dot(self.theta, self.beta)
        t = np.exp(h - np.max(h))
        norm_pmf = t / np.sum(t)

        eps_pmf = np.array([p if p > epsilon else epsilon for p in norm_pmf])
        eps_pmf = eps_pmf - (eps_pmf - epsilon) / \
            np.sum(eps_pmf - epsilon) * np.sum(eps_pmf - norm_pmf)

        return eps_pmf

    def get_pi(self, env, epsilon):
        pmf = np.repeat([self.get_pmf(env, epsilon)], env.n_states, axis=0)

        if env.action_sets is not None:
            for s, a in env.action_sets.items():
                pmf[s, a[1]] = 0
                pmf[s, :] = pmf[s, :] / np.sum(pmf[s, :])

        return pmf

    def update_parameters(self, env, epsilon, delta, a):
        self.theta += delta * \
            (self.beta[:, a] - np.dot(self.beta, self.get_pmf(env, epsilon)))