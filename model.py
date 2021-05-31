import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#DQN
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_hidden = nn.Linear(11, 200) #10=5,400=55,600=60
        self.hidden_out = nn.Linear(200, 3)
        
    def forward(self, x):
        x = F.relu(self.linear_hidden(x))
        x = self.hidden_out(x)
        return x


class Train:
    def __init__(self, model, gamma):
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.RMSprop(model.parameters())#Adam에서 betas,e는 보통 튜닝하지 않음
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        #Bellman Optimality Equation
        #----------------------------------------------------
        pred = self.model(state)

        nn_clone = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            #신경망 수정
            nn_clone[idx][torch.argmax(action[idx]).item()] = Q_new
        #-----------------------------------------------------

        #가중치를 갱신
        self.optimizer.zero_grad()#변화도 버퍼를 0으로 설정
        loss = self.criterion(nn_clone, pred)#손실함수로 loss측정
        loss.backward()#역전파
        self.optimizer.step()#네트워크 업데이트



