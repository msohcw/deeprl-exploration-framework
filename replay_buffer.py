import numpy as np

class ReplayBuffer:
    def __init__(self, size, variables):
        self.map = {}
        self.size = size
        index = 0
        for key, length in variables:
            self.map[key] = (index, index + length)
            index  += length
        self.replay_length = index
        self.memory = np.zeros((self.replay_length, size))

        self.count = 0
        self.index = 0

    def add_replay(self, **kwargs):
        replay = np.zeros(self.replay_length)
        for key in kwargs:
            if key not in self.map: continue
            s, e = self.map[key]
            replay[s:e] = kwargs[key]
        self.memory[:, self.index] = replay
        
        self.count += 1
        self.index += 1
        self.index %= self.size

    def sample(self, batch_size):
        subset_index = np.random.choice(min(self.index, self.size), batch_size, replace = False)
        subset = self.memory[:, subset_index]
        ret = {}
        for key in self.map:
            s, e = self.map[key]
            ret[key] = subset[s:e, :]
        return ret
