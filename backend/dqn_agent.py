"""
Deep Q-Network (DQN) Agent for Smart City Navigation
Implements DQN with experience replay and target network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from typing import List, Tuple, Optional
import json
import os


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture
    Uses CNN to process spatial grid information
    """
    
    def __init__(self, state_shape: Tuple[int, int, int], action_size: int):
        super(DQNetwork, self).__init__()
        
        height, width, channels = state_shape
        
        # Convolutional layers to process spatial information
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after convolutions
        conv_output_size = height * width * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Input shape: (batch, height, width, channels)
        # Reshape to: (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for fully connected layers
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """Experience Replay Buffer for storing and sampling experiences"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with experience replay and target network
    """
    
    def __init__(
        self,
        state_shape: Tuple[int, int, int],
        action_size: int,
        learning_rate: float = 0.0005,  # Reduced from 0.001 for more stable learning
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,  # Increased from 0.01 to keep some exploration
        epsilon_decay: float = 0.997,  # Slower decay (was 0.995)
        buffer_capacity: int = 20000,  # Increased from 10000
        batch_size: int = 128,  # Increased from 64
        target_update_frequency: int = 50,  # Increased from 10 for more stability
        device: Optional[str] = None
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        
        # Device (GPU if available)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Q-Networks
        self.policy_net = DQNetwork(state_shape, action_size).to(self.device)
        self.target_net = DQNetwork(state_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon-greedy)
        
        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Exploitation: choose best action according to policy network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (one gradient update)
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Need enough samples in buffer
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        experiences = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss (Huber loss is more robust than MSE)
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, filepath: str):
        """Save model and training state as compressed ZIP"""
        import zipfile
        import tempfile
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'state_shape': self.state_shape,
            'action_size': self.action_size
        }
        
        # Save to temporary file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            torch.save(checkpoint, tmp_file.name)
            tmp_path = tmp_file.name
        
        # Compress to ZIP
        zip_path = filepath if filepath.endswith('.zip') else filepath + '.zip'
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
            zipf.write(tmp_path, arcname='model.pth')
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Get file sizes for reporting
        original_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        compressed_size = os.path.getsize(zip_path)
        compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        print(f"Model saved to {zip_path}")
        print(f"Compressed size: {compressed_size / (1024*1024):.2f} MB (saved {compression_ratio:.1f}%)")
    
    def load_model(self, filepath: str):
        """Load model and training state from ZIP or regular file"""
        import zipfile
        import tempfile
        
        # Check if file is ZIP
        if filepath.endswith('.zip') or zipfile.is_zipfile(filepath):
            # Extract from ZIP
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_path = tmp_file.name
            
            with zipfile.ZipFile(filepath, 'r') as zipf:
                zipf.extract('model.pth', path=os.path.dirname(tmp_path))
                extracted_path = os.path.join(os.path.dirname(tmp_path), 'model.pth')
                
            checkpoint = torch.load(extracted_path, map_location=self.device, weights_only=False)
            
            # Clean up
            os.remove(extracted_path)
        else:
            # Load regular .pth file
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        
        file_size = os.path.getsize(filepath) / (1024*1024)
        print(f"Model loaded from {filepath} ({file_size:.2f} MB)")
    
    def save_metrics(self, filepath: str):
        """Save training metrics"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def get_stats(self) -> dict:
        """Get training statistics"""
        if not self.episode_rewards:
            return {
                'episodes': 0,
                'avg_reward': 0,
                'max_reward': 0,
                'avg_length': 0,
                'epsilon': self.epsilon
            }
        
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        recent_lengths = self.episode_lengths[-100:]
        
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(recent_rewards),
            'max_reward': np.max(recent_rewards),
            'min_reward': np.min(recent_rewards),
            'avg_length': np.mean(recent_lengths),
            'epsilon': self.epsilon,
            'training_steps': self.training_step
        }


if __name__ == "__main__":
    # Test the DQN agent
    print("Testing DQN Agent")
    print("=" * 50)
    
    state_shape = (15, 15, 3)
    action_size = 5
    
    agent = DQNAgent(state_shape, action_size)
    
    print(f"Policy Network: {agent.policy_net}")
    print(f"\nTotal parameters: {sum(p.numel() for p in agent.policy_net.parameters())}")
    
    # Test forward pass
    dummy_state = np.random.rand(*state_shape)
    action = agent.select_action(dummy_state)
    print(f"\nSelected action: {action}")
    
    # Test experience storage
    next_state = np.random.rand(*state_shape)
    agent.store_experience(dummy_state, action, 1.0, next_state, False)
    print(f"Memory size: {len(agent.memory)}")