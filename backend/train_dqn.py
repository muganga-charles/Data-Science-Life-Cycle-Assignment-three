"""
Training script for DQN agent on Smart City Navigation task
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

from rl_navigation_env import NavigationEnvironment
from dqn_agent import DQNAgent


def train_agent(
    episodes: int = 1000,
    grid_size: int = 15,
    max_steps: int = 300,  # Increased from 200
    save_frequency: int = 100,
    model_dir: str = "models",
    render_frequency: int = 100
):
    """
    Train the DQN agent
    
    Args:
        episodes: Number of training episodes
        grid_size: Size of the grid
        max_steps: Maximum steps per episode
        save_frequency: How often to save the model
        model_dir: Directory to save models
        render_frequency: How often to render episodes
    """
    # Creating directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # Initializing environment and agent
    env = NavigationEnvironment(
        grid_size=grid_size,
        max_steps=max_steps,
        dynamic_traffic=True
    )
    
    state_shape = env.state_shape
    action_size = env.action_space_n
    
    agent = DQNAgent(
        state_shape=state_shape,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,
        batch_size=64,
        target_update_frequency=10
    )
    
    print(f"\n{'='*60}")
    print(f"Training DQN Agent for Smart City Navigation")
    print(f"{'='*60}")
    print(f"Grid Size: {grid_size}x{grid_size}")
    print(f"State Shape: {state_shape}")
    print(f"Action Space: {action_size}")
    print(f"Episodes: {episodes}")
    print(f"Device: {agent.device}")
    print(f"{'='*60}\n")
    
    # Training loop
    best_avg_reward = -float('inf')
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in tqdm(range(episodes), desc="Training"):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Selecting action
            action = agent.select_action(state, training=True)
            
            # Taking step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Storing experience
            agent.store_experience(state, action, reward, next_state, terminated)
            
            # Training agent
            loss = agent.train_step()
            if loss is not None:
                agent.losses.append(loss)
            
            # Updating state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Render occasionally
            if episode % render_frequency == 0 and episode > 0:
                env.render_mode = "human"
                env.render()
                env.render_mode = None
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        
        if terminated:  # Goal reached
            success_count += 1
        
        # Print progress
        if (episode + 1) % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_lengths = episode_lengths[-50:]
            recent_success = sum(1 for i in range(max(0, episode - 49), episode + 1) 
                               if i < len(agent.episode_rewards) 
                               and agent.episode_rewards[i] > 0)
            
            stats = agent.get_stats()
            
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Avg Reward (last 50): {np.mean(recent_rewards):.2f}")
            print(f"Avg Length (last 50): {np.mean(recent_lengths):.2f}")
            print(f"Success Rate (last 50): {recent_success}/50")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Memory Size: {len(agent.memory)}")
            print(f"Training Steps: {agent.training_step}")
            if agent.losses:
                print(f"Avg Loss (last 100): {np.mean(agent.losses[-100:]):.4f}")
            print(f"{'='*60}\n")
        
        # Updating best model (save only best as ZIP)
        if (episode + 1) % 50 == 0:  # Check every 50 episodes
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_model_path = os.path.join(model_dir, "dqn_best.zip")
                agent.save_model(best_model_path)
                print(f"✓ New best model saved! Avg Reward: {avg_reward:.2f}\n")
    
    # Saving final model as ZIP only
    final_model_path = os.path.join(model_dir, "dqn_final.zip")
    agent.save_model(final_model_path)
    
    # Saving metrics
    metrics_path = os.path.join("metrics", f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    agent.save_metrics(metrics_path)
    
    # Plotting training curves
    plot_training_curves(agent.episode_rewards, agent.episode_lengths, agent.losses)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Total Episodes: {episodes}")
    print(f"Total Successes: {success_count}")
    print(f"Success Rate: {success_count/episodes*100:.2f}%")
    print(f"Final Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Best Avg Reward: {best_avg_reward:.2f}")
    print(f"{'='*60}\n")


def plot_training_curves(episode_rewards, episode_lengths, losses):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plotting episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
    if len(episode_rewards) >= 100:
        moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(episode_rewards)), moving_avg, 
                       label='Moving Average (100 episodes)', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plotting episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, label='Episode Length')
    if len(episode_lengths) >= 100:
        moving_avg = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
        axes[0, 1].plot(range(99, len(episode_lengths)), moving_avg,
                       label='Moving Average (100 episodes)', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plotting losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.3, label='Loss')
        if len(losses) >= 100:
            moving_avg = np.convolve(losses, np.ones(100)/100, mode='valid')
            axes[1, 0].plot(range(99, len(losses)), moving_avg,
                           label='Moving Average (100 steps)', linewidth=2)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plotting success rate over time
    window_size = 50
    success_rates = []
    for i in range(window_size, len(episode_rewards) + 1):
        window_rewards = episode_rewards[i-window_size:i]
        success_rate = sum(1 for r in window_rewards if r > 50) / window_size * 100
        success_rates.append(success_rate)
    
    if success_rates:
        axes[1, 1].plot(range(window_size, len(episode_rewards) + 1), success_rates, linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title(f'Success Rate (Moving Window: {window_size})')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to training_curves.png")
    plt.close()


def test_agent(model_path: str, num_episodes: int = 10, grid_size: int = 15, render: bool = True):
    """Test a trained agent"""
    
    # Auto-detect file extension
    if not model_path.endswith(('.zip', '.pth')):
        # Trying ZIP first, then .pth
        if os.path.exists(model_path + '.zip'):
            model_path = model_path + '.zip'
        elif os.path.exists(model_path + '.pth'):
            model_path = model_path + '.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    env = NavigationEnvironment(
        grid_size=grid_size,
        max_steps=200,
        dynamic_traffic=True,
        render_mode="human" if render else None
    )
    
    state_shape = env.state_shape
    action_size = env.action_space_n
    
    agent = DQNAgent(state_shape=state_shape, action_size=action_size)
    agent.load_model(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    print(f"\nTesting Agent: {model_path}")
    print("="*60)
    
    total_rewards = []
    total_lengths = []
    successes = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        if render:
            env.render()
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        
        if terminated:
            successes += 1
            print(f"SUCCESS! Reward: {episode_reward:.2f}, Steps: {episode_length}")
        else:
            print(f"FAILED. Reward: {episode_reward:.2f}, Steps: {episode_length}")
    
    print(f"\n{'='*60}")
    print(f"Test Results")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Successes: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print(f"Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Avg Length: {np.mean(total_lengths):.2f} ± {np.std(total_lengths):.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test DQN agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                       help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--grid_size', type=int, default=15,
                       help='Size of the grid')
    parser.add_argument('--model_path', type=str, default='models/dqn_best.zip',
                       help='Path to model for testing')
    parser.add_argument('--test_episodes', type=int, default=10,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(episodes=args.episodes, grid_size=args.grid_size)
    else:
        test_agent(model_path=args.model_path, num_episodes=args.test_episodes, grid_size=args.grid_size)