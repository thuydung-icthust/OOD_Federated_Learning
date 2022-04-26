import math
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state.cpu(), action, reward, next_state.cpu(), done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.position = 0
        self.buffer.clear()



class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
    
    def get_last_record(self):
        if len(self.states) < 2:
            return None
        else:
            # return in the order of s, a, r, s'
            # print(self.states[-2])
            # print(self.actions[-2])
            # print(self.rewards[-1])
            return self.states[-2], self.actions[-2], self.rewards[-1], self.states[-1]
    
    def act(self, s, a):
        # at state s, agent performs action a
        self.actions.append(a)
        self.states.append(s)
    
    def update(self, r):
        # update reward and next state s' after perform last action a in state s
        # self.states.append(next_s)
        self.rewards.append(r)

    def reset(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()