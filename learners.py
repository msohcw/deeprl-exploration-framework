import random
import math
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Dense

from replay_buffer import ReplayBuffer
from exploration import EpsilonGreedy, CountBasedOptimism, VDBE

EPSILON = 1 * 10 ** -6

class TabularLearner:
    def __init__(self, actions, buckets, limits, terminal_fn, exploration_fn,
                        discount = 0.95, alpha = 0.5, alpha_min = 0.1, alpha_dt = -10 ** -4):
        self.Q = np.random.rand(*buckets, actions)
        self.actions = actions
        self.buckets = buckets
        self.limits = limits
        self.terminal_fn = terminal_fn

        self.discount = discount
        self.eps = eps
        self.eps_dt = eps_dt
        self.alpha = alpha
        self.alpha_dt = alpha_dt
        self.alpha_min = alpha_min

        self.exploration_fn = exploration_fn(self)
        self.exploration_bonus = 0

    def act(self, observation):
        action = self.exploration_fn.explore(observation)
        if type(action) is tuple: # split (action, bonus)
            action, bonus = action 
            self.exploration_bonus = bonus
        self.exploration_fn.modify_eps()
        return action

    def Q(self, s): 
        # abstracts Q(s, a) to allow for general exploration functions
        # returns Q value for given state for all a
        return self.Q[self.discretized(s)] 
    
    def learn(self, s0, s1, reward, action, done, t):
        prev = self.Q[self.discretized(s0) + (action,)]
        now = self.Q[self.discretized(s1)]
        if done: # update should be terminal state update
            update = self.terminal_fn(reward, t)
        else: # update as per normal
            update = self.update_fn(prev, now, reward)
        update += self.exploration_bonus
        self.exploration_bonus = 0
        self.Q[self.discretized(s0) + (action,)] = update
        self.alpha = max(self.alpha_min, self.alpha + self.alpha_dt)

    def update_fn(self, q0_a, q1, r):
        return max(q1) # default, bad update

    def discretized(self, observation):
        b, l = self.buckets, self.limits # shorthand
        # EPSILON used to keep within bucket bounds
        bounded = [min(l[i] - EPSILON,max(-l[i] + EPSILON, observation[i])) for i in range(len(observation))]
        return tuple(math.floor((bounded[i] + l[i]) / 2 / l[i] * b[i]) for i in range(len(bounded)))
        
class TabularQLearner(TabularLearner):
    def update_fn(self, q0_a, q1, r):
        return q0_a + max(self.alpha_min, self.alpha) * (r + self.discount * max(q1) - q0_a)

class TabularSARSALearner(TabularLearner):
    def update_fn(self, q0_a, q1, r):
        expected = max(q1) * (1 - self.eps) + self.eps * sum(q1) / len (q1)
        return q0_a + max(self.alpha_min, self.alpha) * (r + self.discount * expected - q0_a)

class DeepQLearner(TabularLearner):
    MINIMUM_EXPERIENCE = 1000

    def __init__(self, input_dim, layers, batch_size, total_memory, terminal_fn, exploration_fn, 
                    eps = 0.1, eps_dt = -10 * 10 ** -5, discount = 0.95, target_freeze_duration = 2500):
        self.total_memory = total_memory
        self.batch_size = batch_size
        self.terminal_fn = terminal_fn

        stored = (('s0', input_dim), ('s1', input_dim), ('r', 1), ('action', 1), ('done', 1), ('t', 1))
        self.replay_buffer = ReplayBuffer(total_memory, stored)
        
        self.exp_ct = 0
        self.exp_index = 0
        self.model = Sequential()
        self.target_model = Sequential()
        self.input_dim = input_dim
        self.actions = layers[-1]

        self.exploration_fn = exploration_fn(self)
        self.exploration_bonus = 0
        self.discount = discount
        self.target_freeze_duration = target_freeze_duration

        first, layers, end = layers[0], layers[1:-1], layers[-1]
        
        first_layer = Dense(first,
                            batch_input_shape = (None, input_dim),
                            init = 'uniform',
                            activation = 'relu')

        self.model.add(first_layer)
        self.target_model.add(first_layer)
        
        for layer in layers:
            l = Dense(layer, activation = 'relu', init = 'uniform')
            self.model.add(l)
            self.target_model.add(l)
            
        # end with linear to sum to Q-value
        end_layer = Dense(end, activation = 'linear', init = 'uniform')
        self.model.add(end_layer)
        self.target_model.add(end_layer)

        rms = keras.optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer = rms, loss = 'mse')
        self.target_model.compile(optimizer = rms, loss = 'mse')

    def learn(self, s0, s1, reward, action, done, t):
        reward += self.exploration_bonus
        self.exploration_bonus = 0
        self.replay_buffer.add_replay(s0=s0, s1=s1, r=reward, action=action, done=done, t=t)

        if self.replay_buffer.index % self.target_freeze_duration == 0:
            print("Copied to target network")
            self.target_model.set_weights(self.model.get_weights())

        if self.replay_buffer.count > DeepQLearner.MINIMUM_EXPERIENCE: 
            self.experience_replay()

    def experience_replay(self):
        replays = self.replay_buffer.sample(self.batch_size)
        s0 = replays.get('s0')
        s1 = replays.get('s1')
        r = replays.get('r').flatten()
        action = replays.get('action').flatten()
        done = replays.get('done').flatten()
        t = replays.get('t').flatten()

        s0_q_values = self.model.predict(s0.T) 
        s1_q_values = self.future_fn(s1)
        
        td_error = []

        # update Q values
        for k, q in enumerate(s0_q_values):
            a = int(action[k])
            if bool(done[k]):
                q_1 = self.terminal_fn(r[k], t[k])
            else:
                q_1 = r[k] + self.discount * s1_q_values[k]
            td_error.append(q_1 - q[a])
            q[a] = q_1

        td_error = np.array(td_error)

        if type(self.exploration_fn) == VDBE:
            self.exploration_fn.modify_eps(s0, td_error)
        
        loss = self.model.train_on_batch(s0.T, s0_q_values)

    def future_fn(self, s1):
        return np.amax(self.target_model.predict(s1.T), axis = 1)

    def Q(self, s):
        return self.model.predict(s.reshape(1,self.input_dim))

class DoubleDeepQLearner(DeepQLearner):
    def future_fn(self, s1):
        # use model to pick actions, target to evaluate Q(s, a)
        actions = np.argmax(self.model.predict(s1.T), axis = 1)
        q_values = self.target_model.predict(s1.T)
        # takes the element of each row according to actions
        return np.diagonal(np.take(q_values, actions, axis = 1))
