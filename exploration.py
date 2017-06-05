import numpy as np
import math, random

class EpsilonGreedy:
    def __init__(self, learner, eps, eps_dt):
        self.learner = learner
        self.eps = eps
        self.eps_dt = eps_dt

    def explore(self, observation):
        if random.random() < self.eps: # explore
            action = math.floor(random.random() * self.learner.actions)
        else: # exploit
            action = np.argmax(self.learner.Q(observation))
        return action

    def modify_eps(self):
        self.eps = max(0, self.eps + self.eps_dt)

class CountBasedOptimism(EpsilonGreedy):
    def __init__(self, learner, eps, eps_dt, hash_fn, exploration_bonus):
        super().__init__(learner, eps, eps_dt)
        self.hash_fn = hash_fn
        self.exploration_bonus = exploration_bonus
        self.counter = {}

    def explore(self, observation):
        hashed = self.hash_fn(observation)
        self.counter[hashed] = self.counter.get(hashed, 0) + 1
        bonus = self.exploration_bonus / (self.counter[hashed] ** 0.5)
        return (super().explore(observation), bonus)

class VDBE(EpsilonGreedy):
    def __init__(self, learner, eps, hash_fn, sigma, delta):
        super().__init__(learner, eps, 0)
        self.hash_fn = hash_fn
        self.state_eps = {}
        self.base_eps = eps
        self.sigma = sigma
        self.delta = delta

    def explore(self, observation):
        hashed = self.hash_fn(observation)
        self.eps = self.state_eps.get(hashed, self.base_eps)
        return super().explore(observation)

    def modify_eps(self, *args):
        if len(args) == 2:
            states, td_error = args
            f = -abs(td_error) / self.sigma
            f = (1-np.exp(f)) / (1+np.exp(f))
            for k, state in enumerate(states.T):
                hashed = self.hash_fn(state)
                update = self.delta * f[k] + (1-self.delta) * self.state_eps.get(hashed, self.base_eps)
                self.state_eps[hashed] = update
