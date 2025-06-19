import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.atari_configs import TD0_CONFIG

class ValueNetwork(nn.Module):
    def __init__(self, input_shape):
        super(ValueNetwork, self).__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # scalar output
        )

    def forward(self, x):
        x = x / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class DeepTD0Agent:
    def __init__(self, input_shape, num_actions, config=TD0_CONFIG, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = config['gamma']
        self.lr = config['lr']
        self.num_actions = num_actions

        self.value_net = ValueNetwork(input_shape).to(self.device)
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

    def select_action(self, state):
        # Random policy baseline
        return np.random.randint(0, self.num_actions)

    def update(self, state, reward, next_state, done):
        # Prepare tensors
        state_t = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
        next_state_t = torch.tensor(np.array([next_state]), dtype=torch.float32, device=self.device)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)

        # Compute value estimates
        v_s = self.value_net(state_t)
        with torch.no_grad():
            v_s_next = self.value_net(next_state_t)

        # Build TD(0) target as tensor
        target = reward_t + (1.0 - done_t) * self.gamma * v_s_next

        # Compute loss and update
        loss = nn.functional.mse_loss(v_s, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
