import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import rl_utils

class PolicyNet(torch.nn.Module):       #策略网络，输入为某个状态，输出为该状态的概率分布
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AC:
    def __init__(self,state_dim, hidden_dim, action_dim, a_lr,c_lr, gamma, device):
        self.policy_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.Q_net = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=a_lr)  # 定义Adam优化器
        self.optimizer_Q = torch.optim.Adam(self.Q_net.parameters(), lr=c_lr)  # 定义Adam优化器
        self.gamma = gamma
        self.device = device

    def Actor(self, state):  # 根据动作概率分布随机采样，根据状态得到动作
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        action_dist = torch.distributions.Categorical(probs)  # 构建一个概率分布函数
        action = action_dist.sample()  # 从概率分布函数中抽样
        try:
            return action.item()
        except:
            return action

    def Critic(self,state,action):      #根据状态动作得到Q值
        return self.Q_net(state).gather(1, action)


    def update(self, transition_dict):
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        next_actions = self.Actor(transition_dict['next_states']).view(-1,1)

        #更新价值函数
        Q_now = self.Critic(states,actions)
        Q_target = (rewards + self.gamma*self.Critic(next_states,next_actions))*(1-dones)
        Q_loss = torch.mean(F.mse_loss(Q_now,Q_target))
        self.optimizer_Q.zero_grad()
        Q_loss.backward()
        self.optimizer_Q.step()

        #更新策略函数
        log_probs = torch.log(self.policy_net(states).gather(1, actions))
        policy_loss = torch.mean(-log_probs * (Q_target-Q_now).detach())
        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()



a_lr = 1e-3
c_lr = 1e-2
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = AC(state_dim, hidden_dim, action_dim, a_lr,c_lr, gamma,device)

return_list = []
for i in range(100):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state = env.reset(seed=0)[0]
            done = False            #表示还未完成
            transition_dict = {'states':[],'actions':[],'next_states':[],'rewards':[],'dones':[]}
            while not done:
                action = agent.Actor(state)
                next_state, reward, done, _,_ = env.step(action)
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return = reward + episode_return
            agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)