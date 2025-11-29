# ============================================
# File: shared/env.py
# ============================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from .config import (
    GRID_SIZE,
    MAX_STEPS_PER_EPISODE,
    N_GOOD_TILES,
    N_BAD_TILES,
    GOOD_TILE_REWARD,
    BAD_TILE_PENALTY,
    WALL_PENALTY,
    STEP_PENALTY,
    FRAME_STACK_SIZE,
)


class ManualFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=FRAME_STACK_SIZE):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(c * num_stack, h, w),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.concatenate(list(self.frames), axis=0)


class HallOfMirrorsGridworld(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size=GRID_SIZE,
        max_steps=MAX_STEPS_PER_EPISODE,
        n_good_tiles=N_GOOD_TILES,
        n_bad_tiles=N_BAD_TILES,
        seed=0,
        # Granular Randomization Flags
        random_rot=False,
        random_step=False,
        random_val=False,
        # Fixed Defaults
        fixed_sensor_rotation=0,
        fixed_step_size=1,
        fixed_good_is_red=False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_good_tiles = n_good_tiles
        self.n_bad_tiles = n_bad_tiles
        self._np_seed = seed
        self.rng = np.random.RandomState(seed)

        # Config
        self.random_rot = random_rot
        self.random_step = random_step
        self.random_val = random_val
        
        self.fixed_sensor_rotation = fixed_sensor_rotation
        self.fixed_step_size = fixed_step_size
        self.fixed_good_is_red = fixed_good_is_red

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5, grid_size, grid_size),
            dtype=np.float32,
        )

        self.grid = None
        self.agent_pos = None
        self.steps = 0
        self.sensor_rotation = 0
        self.step_size = 1
        self.good_is_red = False
        self.irrelevant_pattern = None
        self.collected = None

    def reseed(self, seed: int):
        self._np_seed = seed
        self.rng = np.random.RandomState(seed)

    def _sample_gauges(self):
        # 1. Rotation
        if self.random_rot:
            self.sensor_rotation = int(self.rng.choice([0, 1, 2, 3]))
        else:
            self.sensor_rotation = int(self.fixed_sensor_rotation)
            
        # 2. Step Size
        if self.random_step:
            self.step_size = int(self.rng.choice([1, 2]))
        else:
            self.step_size = int(self.fixed_step_size)
            
        # 3. Value Map
        if self.random_val:
            self.good_is_red = bool(self.rng.choice([0, 1]))
        else:
            self.good_is_red = bool(self.fixed_good_is_red)

        self.irrelevant_pattern = self.rng.rand(self.grid_size, self.grid_size).astype(
            np.float32
        )

    def _generate_layout(self):
        g = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for _ in range(self.grid_size * 2):
            y = self.rng.randint(0, self.grid_size)
            x = self.rng.randint(0, self.grid_size)
            g[y, x] = 1  # walls

        empty = list(zip(*np.where(g == 0)))
        self.rng.shuffle(empty)

        n_avail = max(0, len(empty) - 1)
        n_g = min(self.n_good_tiles, n_avail)
        n_b = min(self.n_bad_tiles, n_avail - n_g)

        count = 0
        for y, x in empty:
            if count < n_g:
                g[y, x] = 2
            elif count < n_g + n_b:
                g[y, x] = 3
            else:
                break
            count += 1

        empty = list(zip(*np.where(g == 0)))
        if not empty:
            cy = self.grid_size // 2
            cx = self.grid_size // 2
            g[cy, cx] = 0
            empty = [(cy, cx)]
        self.agent_pos = list(empty[self.rng.randint(0, len(empty))])
        self.grid = g
        self.collected = np.zeros_like(g, dtype=bool)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.reseed(seed)
        self.steps = 0
        self._sample_gauges()
        self._generate_layout()
        return self._get_obs(), {}

    def _rotate_obs(self, obs):
        k = self.sensor_rotation
        if k == 0:
            return obs
        obs_hw_c = np.transpose(obs, (1, 2, 0))
        obs_rot = np.rot90(obs_hw_c, k=k, axes=(0, 1))
        return np.transpose(obs_rot, (2, 0, 1))

    def _get_obs(self):
        wall = (self.grid == 1).astype(np.float32)
        red = (self.grid == 2).astype(np.float32)
        blue = (self.grid == 3).astype(np.float32)
        agent = np.zeros_like(wall, dtype=np.float32)
        ay, ax = self.agent_pos
        agent[ay, ax] = 1.0
        obs = np.stack([wall, red, blue, agent, self.irrelevant_pattern], axis=0)
        return self._rotate_obs(obs)

    def step(self, action):
        self.steps += 1
        reward = 0.0

        dy, dx = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[action]

        for _ in range(self.step_size):
            ny = self.agent_pos[0] + dy
            nx = self.agent_pos[1] + dx
            if (
                0 <= ny < self.grid_size
                and 0 <= nx < self.grid_size
                and self.grid[ny, nx] != 1
            ):
                self.agent_pos = [ny, nx]
            else:
                reward += WALL_PENALTY
                break

        ay, ax = self.agent_pos
        if self.grid[ay, ax] in (2, 3) and not self.collected[ay, ax]:
            self.collected[ay, ax] = True
            is_red = self.grid[ay, ax] == 2
            if self.good_is_red:
                tile_reward = GOOD_TILE_REWARD if is_red else BAD_TILE_PENALTY
            else:
                tile_reward = BAD_TILE_PENALTY if is_red else GOOD_TILE_REWARD
            reward += tile_reward

            self.grid[ay, ax] = 0

        reward += STEP_PENALTY
        truncated = self.steps >= self.max_steps
        terminated = False
        return self._get_obs(), reward, terminated, truncated, {}
    
    def force_noise_pattern(self, seed: int):
        """For analysis only: overrides the noise channel."""
        rng = np.random.RandomState(seed)
        self.irrelevant_pattern = rng.rand(self.grid_size, self.grid_size).astype(
            np.float32
        )