# agents/dqn_agent.py

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.atari_configs import DQN_CONFIG

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # state, next_state: numpy arrays; action, reward, done: scalars
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(QNetwork, self).__init__()
        c, h, w = input_shape
        # Define convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Compute conv output shape
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # normalize pixel intensities
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class DQNAgent:
    def __init__(self, input_shape, num_actions, config=DQN_CONFIG, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_actions = num_actions
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_start']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay_steps = config['epsilon_decay_steps']
        self.epsilon_decay = (self.epsilon_start - self.epsilon_end) / self.epsilon_decay_steps

        # Networks
        self.policy_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net = QNetwork(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['lr'])

        # Replay buffer
        self.memory = ReplayBuffer(config['buffer_size'])
        self.batch_size = config['batch_size']
        self.target_update_freq = config['target_update_freq']
        self.learn_steps = 0
        self.update_freq = config.get("update_freq", 4)

    def select_action(self, state):
        # state: numpy array shape (c,h,w)
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def push_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        if self.learn_steps % self.update_freq != 0:
            self.learn_steps += 1
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute current Q values
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q_values = rewards_t + (self.gamma * next_q_values * (1 - dones_t))

        # Compute loss
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay

        # Update target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
