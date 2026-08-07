import random
import time
from typing import Dict
import numpy as np
import pygame
from utility import play_q_table
from cat_env import make_env
import math
#############################################################################
# TODO: YOU MAY ADD ADDITIONAL IMPORTS OR FUNCTIONS HERE.                   #
#############################################################################


def get_manhattan_dist(state: int) -> int:
    """Extracts positions from state integer (RCrc) and computes distance."""
    # State format: bot_row, bot_col, cat_row, cat_col
    bot_r = state // 1000
    bot_c = (state // 100) % 10
    cat_r = (state // 10) % 10
    cat_c = state % 10
    return abs(bot_r - cat_r) + abs(bot_c - cat_c)





#############################################################################
# END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
#############################################################################

def train_bot(cat_name, render: int = -1):
    env = make_env(cat_type=cat_name)
    
    # Initialize Q-table with all possible states (0-9999)
    # Initially, all action values are zero.
    q_table: Dict[int, np.ndarray] = {
        state: np.zeros(env.action_space.n) for state in range(10000)
    }

    # Training hyperparameters
    episodes = 5000 # Training is capped at 5000 episodes for this project
    
    #############################################################################
    # TODO: YOU MAY DECLARE OTHER VARIABLES AND PERFORM INITIALIZATIONS HERE.   #
    #############################################################################
    # Hint: You may want to declare variables for the hyperparameters of the    #
    # training process such as learning rate, exploration rate, etc.            #
    #############################################################################
    
    # Training hyperparameters
    learning_rate = 0.15        # (alpha) How fast Q-values update
    discount_factor = 0.95      # (gamma) Importance of future rewards
    exploration_rate = 1.0      # (epsilon) Initial random move chance (100%)
    min_exploration_rate = 0.01 # Minimum random move chance (1%)
    decay_rate = 0.001          # How fast exploration_rate decreases each episode









    
    #############################################################################
    # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
    #############################################################################
    
    for ep in range(1, episodes + 1):
        ##############################################################################
        # TODO: IMPLEMENT THE Q-LEARNING TRAINING LOOP HERE.                         #
        ##############################################################################
        # Hint: These are the general steps you must implement for each episode.     #
        # 1. Reset the environment to start a new episode.                           #
        # 2. Decide whether to explore or exploit.                                   #
        # 3. Take the action and observe the next state.                             #
        # 4. Since this environment doesn't give rewards, compute reward manually    #
        # 5. Update the Q-table accordingly based on agent's rewards.                #
        ############################################################################## 


        # 1. Reset the environment for a new episode
        obs, _ = env.reset()
        done = False
        current_distance = get_manhattan_dist(obs)

        # Run until CatBot catches the cat or episode ends
        while not done:
            # 2. Choose Action: Explore (random) vs Exploit (best Q-value)
            if random.random() < exploration_rate:
                action = env.action_space.sample()  # Random move
            else:
                action = int(np.argmax(q_table[obs]))  # Best learned move

            # 3. Take action and observe the next state
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_distance = get_manhattan_dist(next_obs)

            # 4. Compute Custom Reward
            if terminated:
                reward = 100.0  # Big reward for catching the cat
            else:
                # Reward moving closer, penalize moving further away
                distance_change = current_distance - next_distance
                # Small step penalty (-0.1) to encourage speed
                reward = (distance_change * 2.0) - 0.1

            # 5. Update Q-Table using Q-Learning Formula
            best_next_action = np.argmax(q_table[next_obs])
            target_q = reward + discount_factor * q_table[next_obs][best_next_action] * (1 - int(terminated))
            q_error = target_q - q_table[obs][action]
            
            # Apply update
            q_table[obs][action] += learning_rate * q_error

            # Advance state and update distance for next iteration
            obs = next_obs
            current_distance = next_distance

        # Decay exploration rate at the end of each episode
        exploration_rate = max(min_exploration_rate, exploration_rate * math.exp(-decay_rate))
        






























        
        
        #############################################################################
        # END OF YOUR CODE. DO NOT MODIFY ANYTHING BEYOND THIS LINE.                #
        #############################################################################

        # If rendering is enabled, play an episode every 'render' episodes
        if render != -1 and (ep == 1 or ep % render == 0):
            viz_env = make_env(cat_type=cat_name)
            play_q_table(viz_env, q_table, max_steps=100, move_delay=0.02, window_title=f"{cat_name}: Training Episode {ep}/{episodes}")
            print('episode', ep)

    return q_table