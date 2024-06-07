import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from tqdm import tqdm

# Define the MsPacman environment with render_mode
env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
env = AtariPreprocessing(env, frame_skip=1)
env = FrameStack(env, 4)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'You are using {device}')

buffer_size = 100000
batch_size = 64
tau = 0.001

def normalize(state):
    return np.array(state) / 255.0

class QNetwork(nn.Module):
    """Neural Network for approximating Q-values."""
    def __init__(self, state_shape, action_size):
        super(QNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_input_size = self._get_conv_output(state_shape)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
    
    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))
    
    def forward(self, state):
        x = self.conv_layers(state)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)


    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).float().to(device)
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters())
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    def act(self, state, eps=0.):
        state = torch.from_numpy(normalize(state)).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Compute Q targets for next states
        q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (gamma * q_target_next * (1 - dones))

        # Compute Q expected for current states
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_target)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


state_size = env.observation_space.shape
action_size = env.action_space.n

agent = DQNAgent(state_size=state_size, action_size=action_size)

def dqn(n_episodes=100, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()  # Unpack the state from the reset method
        state = normalize(state)
        score = 0
        lives = env.unwrapped.ale.lives()  # Get the initial number of lives

        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)  # Unpack the state from the step method
            next_state = normalize(next_state)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            # Check if lives have changed
            if env.unwrapped.ale.lives() < lives:
                lives = env.unwrapped.ale.lives()  # Update the number of lives
                if lives == 0:  # If all lives are lost, break the loop
                    done = True
                else:
                    done = False
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print(f'Episode {i_episode}, Avg Score: {np.mean(scores_window)}')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 1000.0:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window)}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores, steps

scores = dqn()

def plot_scores(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    fig.savefig('training_scores.png')

plot_scores(scores)

def create_video(agent, env, filename="mspacman.mp4"):
    video_env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: True, video_length=0)
    state, _ = video_env.reset()
    state = normalize(state)
    done = False
    while not done:
        action = agent.act(state, eps=0.0)  # Use greedy policy for evaluation
        state, reward, done, _, _ = video_env.step(action)
        state = normalize(state)
    video_env.close()
    video_dir = './video'
    video_file = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][0]
    os.rename(os.path.join(video_dir, video_file), filename)

create_video(agent, env)