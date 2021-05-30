import torch
import random
from collections import deque
from model import NN, Train


MAX_MEMORY = 100000
BATCH_SIZE = 1000


class Agent:

    def __init__(self):
        self.n_games = 0
        self.random_game = 50 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = NN()
        self.trainer = Train(self.model, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self): #게임이 끝났을때 모델을 훈련
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done): #게임중에 모델을 훈련
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.random_game =70-self.n_games # 게임을 할수록 무작위성이 줄어듬
        action = [0,0,0]
        if random.randint(0, 200) < self.random_game:#게임횟수 50회부터는 학습한 모델로 예측 
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # [-0.0144,0.0325,-0.0941]
            move = torch.argmax(prediction).item()# 가장 큰 예측값의 인덱스를 반환함(argmax->tensor(idx))이걸 .item()으로 idx만 남게 함
            action[move] = 1

        return action
