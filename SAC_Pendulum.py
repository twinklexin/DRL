import torch
import gymnasium as gym
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
import rl_utils
from tqdm import tqdm
import matplotlib.pyplot as plt

class PolicyNet(nn.Module):
    def __init__(self,state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self,x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
        action = action * self.action_bound
        return action, log_prob

class QvalueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QvalueNet, self).__init__()
        self.fc11 = nn.Linear(state_dim + action_dim,hidden_dim)
        self.fc12 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out1 = nn.Linear(hidden_dim, 1)

        self.fc21 = nn.Linear(state_dim + action_dim,hidden_dim)
        self.fc22 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, a):
        x  = torch.cat([x,a], dim=1)

        # Q1
        x1 = F.relu(self.fc11(x))
        x1 = F.relu(self.fc12(x1))
        Q1 = self.fc_out1(x1)
        # Q2
        x2 = F.relu(self.fc21(x))
        x2 = F.relu(self.fc22(x2))
        Q2 = self.fc_out2(x2)

        return Q1, Q2


class SAC:
    def __init__(self, state_dim,  hidden_dim, action_dim, action_bound, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = QvalueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_target = QvalueNet(state_dim, hidden_dim, action_dim).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer= torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = critic_lr)
        
        self.log_alpha = torch.tensor(np.log(0.01), dtype = torch.float).to(device)
        self.log_alpha_requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr = alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau 
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype = torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype = torch.float).view(-1,1).to(self.device)

        #更新价值网络
        with torch.no_grad():
            next_actions, next_log_probs = self.actor(next_states)
            entropy = -next_log_probs
            Q1_target, Q2_target = self.critic_target(next_states, next_actions)
            next_values = torch.min(Q1_target, Q2_target) + self.log_alpha * entropy
            td_target = rewards + self.gamma * next_values * (1 - dones)

        Q1, Q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(Q1, td_target) + F.mse_loss(Q2, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #更新策略网络
        new_actions, log_probs = self.actor(states)
        entropy = -log_probs
        Q1_target, Q2_target = self.critic_target(next_states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(Q1_target, Q2_target))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #更新alpha
        alpha_loss = torch.mean(self.log_alpha.exp() * (entropy - self.target_entropy).detach())
        alpha_loss.requires_grad = True
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        for param_target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)


env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # 动作最大值
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 200
hidden_dim = 128
gamma = 0.99
tau = 0.005  # 软更新参数
buffer_size = 100000
minimal_size = 1000
batch_size = 64
target_entropy = -env.action_space.shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SAC(state_dim, hidden_dim, action_dim, action_bound,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return = 0
            state = env.reset()[0]
            done = False
            ep = 0
            while not done:
                ep += 1
                action = agent.take_action(state)
                next_state, reward, done, _, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'rewards': b_r, 'next_states': b_ns, 'dones': b_d}
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

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format('Pendulum-v1'))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DDPG on {}'.format('Pendulum-v1'))
plt.show()