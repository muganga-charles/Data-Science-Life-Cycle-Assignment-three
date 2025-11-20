"""
Smart City Navigation Environment for Reinforcement Learning
Implements a Gym-style environment for training autonomous navigation agents
"""

import numpy as np
import random
from typing import Tuple, Dict, List, Optional
from enum import IntEnum
from collections import deque


class CellType(IntEnum):
    """Types of cells in the city grid"""
    STREET = 0
    HIGHWAY = 1
    RESIDENTIAL = 2
    OBSTACLE = 9


class TrafficLevel(IntEnum):
    """Traffic congestion levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    BLOCKED = 9


class NavigationEnvironment:
    """
    Smart City Navigation Environment
    
    State: (agent_x, agent_y, goal_x, goal_y, traffic_grid)
    Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
    Rewards: Negative for time, positive for reaching goal, penalties for traffic/obstacles
    """
    
    def __init__(
        self, 
        grid_size: int = 15,
        max_steps: int = 200,
        dynamic_traffic: bool = True,
        render_mode: Optional[str] = None
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.dynamic_traffic = dynamic_traffic
        self.render_mode = render_mode
        
        # Action space: Up, Down, Left, Right, Stay
        self.action_space_n = 5
        
        # State space dimensions
        self.state_shape = (grid_size, grid_size, 3)  # position, goal, traffic
        
        # Initialize environment
        self.city_grid = None
        self.traffic_grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.steps_taken = 0
        self.episode = 0
        
        # Traffic patterns (for rush hour simulation)
        self.time_step = 0
        self.rush_hour_start = 50
        self.rush_hour_end = 100
    
    def _has_valid_path(self, start, goal):
        from collections import deque

        queue = deque([tuple(start)])
        visited = set([tuple(start)])
        gx, gy = goal

        while queue:
            x, y = queue.popleft()
            if (x, y) == (gx, gy):
                return True

            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                nx, ny = x + dx, y + dy

                # Inside grid and not obstacle
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.city_grid[nx, ny] != 9 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        return False


        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state and guarantee reachable path"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.episode += 1
        self.steps_taken = 0
        self.time_step = 0

        self._generate_city_layout()
        self._initialize_traffic()

        self.agent_pos = self._get_random_valid_position()
        self.goal_pos = self._get_random_valid_position()

        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = self._get_random_valid_position()

        attempts = 0
        while not self._has_valid_path(self.agent_pos, self.goal_pos):
            attempts += 1

            self.agent_pos = self._get_random_valid_position()
            self.goal_pos = self._get_random_valid_position()

            # Ensure start != goal
            while np.array_equal(self.agent_pos, self.goal_pos):
                self.goal_pos = self._get_random_valid_position()

            # If too many failed attempts → regenerate whole map
            if attempts > 40:
                self._generate_city_layout()
                self._initialize_traffic()
                attempts = 0

        # Everything is valid
        state = self._get_state()
        info = self._get_info()

        return state, info

    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Returns:
            state: Current state
            reward: Reward for this step
            terminated: Whether episode ended (goal reached or collision)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        self.steps_taken += 1
        self.time_step += 1
        
        # Update dynamic traffic
        if self.dynamic_traffic and self.steps_taken % 10 == 0:
            self._update_traffic()
        
        # Execute action
        old_pos = self.agent_pos.copy()
        new_pos = self._apply_action(action)
        
        # Calculate reward
        reward, terminated = self._calculate_reward(old_pos, new_pos, action)
        
        # Update agent position if valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        
        # Check if episode is truncated (max steps reached)
        truncated = self.steps_taken >= self.max_steps
        
        state = self._get_state()
        info = self._get_info()
        
        return state, reward, terminated, truncated, info
    
    def _generate_city_layout(self):
        """Generate a realistic city layout with roads and buildings"""
        self.city_grid = np.full((self.grid_size, self.grid_size), CellType.STREET, dtype=int)
        
        # Create highways (faster roads)
        # Horizontal highways
        highway_rows = [self.grid_size // 3, 2 * self.grid_size // 3]
        for row in highway_rows:
            if row < self.grid_size:
                self.city_grid[row, :] = CellType.HIGHWAY
        
        # Vertical highways
        highway_cols = [self.grid_size // 3, 2 * self.grid_size // 3]
        for col in highway_cols:
            if col < self.grid_size:
                self.city_grid[:, col] = CellType.HIGHWAY
        
        # Add some residential areas (slower) - reduced from 3 to 2
        for _ in range(2):
            x = random.randint(0, self.grid_size - 3)
            y = random.randint(0, self.grid_size - 3)
            # Don't place on highways
            if self.city_grid[x, y] != CellType.HIGHWAY:
                self.city_grid[x:x+2, y:y+2] = CellType.RESIDENTIAL
        
        # Add random obstacles (buildings, construction) - REDUCED significantly
        num_obstacles = int(self.grid_size * 0.15)  # Reduced from 0.3 to 0.15
        for _ in range(num_obstacles):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            # Don't place obstacles on highways
            if self.city_grid[x, y] != CellType.HIGHWAY:
                self.city_grid[x, y] = CellType.OBSTACLE
    
    def _initialize_traffic(self):
        """Initialize traffic conditions"""
        self.traffic_grid = np.random.choice(
            [TrafficLevel.LOW, TrafficLevel.MEDIUM], 
            size=(self.grid_size, self.grid_size),
            p=[0.85, 0.15]  # Changed from [0.7, 0.3] to have less initial traffic
        )
        
        # No traffic on obstacles
        self.traffic_grid[self.city_grid == CellType.OBSTACLE] = 0
    
    def _update_traffic(self):
        """Update traffic conditions dynamically (simulate rush hour)"""
        # Check if in rush hour
        is_rush_hour = self.rush_hour_start <= self.time_step <= self.rush_hour_end
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.city_grid[i, j] == CellType.OBSTACLE:
                    continue
                
                # Higher probability of traffic during rush hour - but less aggressive
                if is_rush_hour:
                    if random.random() < 0.15:  # Reduced from 0.3
                        self.traffic_grid[i, j] = min(
                            TrafficLevel.HIGH, 
                            self.traffic_grid[i, j] + 1
                        )
                else:
                    if random.random() < 0.25:  # Increased from 0.2 to reduce traffic faster
                        self.traffic_grid[i, j] = max(
                            TrafficLevel.LOW,
                            self.traffic_grid[i, j] - 1
                        )
    
    def _apply_action(self, action: int) -> np.ndarray:
        """Apply action and return new position"""
        x, y = self.agent_pos
        
        if action == 0:  # Up
            return np.array([x - 1, y])
        elif action == 1:  # Down
            return np.array([x + 1, y])
        elif action == 2:  # Left
            return np.array([x, y - 1])
        elif action == 3:  # Right
            return np.array([x, y + 1])
        else:  # Stay (action == 4)
            return np.array([x, y])
    
    def _is_valid_position(self, pos: np.ndarray) -> bool:
        """Check if position is valid (in bounds and not obstacle)"""
        x, y = pos
        
        # Check bounds
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        
        # Check obstacle
        if self.city_grid[x, y] == CellType.OBSTACLE:
            return False
        
        return True
    
    def _get_random_valid_position(self) -> np.ndarray:
        """Get a random valid position"""
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            pos = np.array([x, y])
            if self._is_valid_position(pos):
                return pos
    
    def _calculate_reward(
        self, 
        old_pos: np.ndarray, 
        new_pos: np.ndarray, 
        action: int
    ) -> Tuple[float, bool]:
        """Calculate reward for the action taken"""
        terminated = False
        
        # Goal reached - large positive reward
        if np.array_equal(new_pos, self.goal_pos):
            reward = 200.0  # Increased from 100
            terminated = True
            return reward, terminated
        
        # Invalid move (obstacle or out of bounds)
        if not self._is_valid_position(new_pos):
            reward = -5.0  # Reduced penalty (was -10)
            return reward, terminated
        
        # Base reward - very small penalty for each step
        reward = -0.1  # Reduced from -1.0 to encourage exploration
        
        # Traffic penalty - reduced impact
        x, y = new_pos
        traffic = self.traffic_grid[x, y]
        
        if traffic == TrafficLevel.LOW:
            reward -= 0.2  # Reduced from 0.5
        elif traffic == TrafficLevel.MEDIUM:
            reward -= 0.5  # Reduced from 2.0
        elif traffic == TrafficLevel.HIGH:
            reward -= 1.0  # Reduced from 5.0
        
        # Road type bonus/penalty - increased highway bonus
        road_type = self.city_grid[x, y]
        if road_type == CellType.HIGHWAY:
            reward += 1.0  # Increased from 0.5
        elif road_type == CellType.RESIDENTIAL:
            reward -= 0.3  # Reduced from 1.0
        
        # Distance-based reward (MOST IMPORTANT - helps guide the agent)
        old_dist = np.linalg.norm(old_pos - self.goal_pos)
        new_dist = np.linalg.norm(new_pos - self.goal_pos)
        
        distance_improvement = old_dist - new_dist
        
        if distance_improvement > 0:
            # Reward for moving closer - scaled by improvement
            reward += 2.0 * distance_improvement  # Increased from 1.0
        else:
            # Small penalty for moving away
            reward -= 0.3  # Reduced from 0.5
        
        # Additional reward for being close to goal (helps final approach)
        if new_dist < 5.0:
            reward += (5.0 - new_dist) * 0.5  # Bonus when very close
        
        # Penalty for staying in place unnecessarily
        if action == 4 and not np.array_equal(new_pos, self.goal_pos):
            reward -= 1.0  # Reduced from 2.0
        
        return reward, terminated
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state representation
        Returns a 3-channel grid: [position_channel, goal_channel, traffic_channel]
        """
        state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Channel 0: Agent position (1 at agent location)
        state[self.agent_pos[0], self.agent_pos[1], 0] = 1.0
        
        # Channel 1: Goal position (1 at goal location)
        state[self.goal_pos[0], self.goal_pos[1], 1] = 1.0
        
        # Channel 2: Normalized traffic grid
        state[:, :, 2] = self.traffic_grid / 9.0  # Normalize to [0, 1]
        
        return state
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        return {
            'agent_pos': self.agent_pos.tolist(),
            'goal_pos': self.goal_pos.tolist(),
            'steps_taken': self.steps_taken,
            'manhattan_distance': int(np.sum(np.abs(self.agent_pos - self.goal_pos))),
            'is_rush_hour': self.rush_hour_start <= self.time_step <= self.rush_hour_end,
            'episode': self.episode
        }
    
    def get_grid_state(self) -> Dict:
        """Get complete grid state for visualization"""
        return {
            'city_grid': self.city_grid.tolist(),
            'traffic_grid': self.traffic_grid.tolist(),
            'agent_pos': self.agent_pos.tolist(),
            'goal_pos': self.goal_pos.tolist(),
            'grid_size': self.grid_size
        }
    
    def render(self):
        """Render the environment (for debugging)"""
        if self.render_mode == "human":
            print(f"\nStep {self.steps_taken}/{self.max_steps}")
            print(f"Agent: {self.agent_pos}, Goal: {self.goal_pos}")
            print(f"Distance to goal: {np.sum(np.abs(self.agent_pos - self.goal_pos))}")
            
            # Create visualization grid
            vis_grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
            vis_grid[:] = '.'
            
            # Mark obstacles
            vis_grid[self.city_grid == CellType.OBSTACLE] = '#'
            
            # Mark highways
            vis_grid[self.city_grid == CellType.HIGHWAY] = '='
            
            # Mark goal
            vis_grid[self.goal_pos[0], self.goal_pos[1]] = 'G'
            
            # Mark agent
            vis_grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
            
            # Print grid
            for row in vis_grid:
                print(' '.join(row))
            print()


if __name__ == "__main__":
    # Test the environment
    env = NavigationEnvironment(grid_size=10, render_mode="human")
    
    print("Testing Navigation Environment")
    print("=" * 50)
    
    state, info = env.reset(seed=42)
    print(f"Initial state shape: {state.shape}")
    print(f"Initial info: {info}")
    
    env.render()
    
    # Run a few random steps
    for i in range(5):
        action = random.randint(0, 4)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Done={terminated or truncated}")
        env.render()
        
        if terminated or truncated:
            print("Episode finished!")
            break