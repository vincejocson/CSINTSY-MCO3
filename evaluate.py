import numpy as np
from training import train_bot
from cat_env import make_env

# List of open cats to evaluate
cat_names = ['batmeow', 'mittens', 'paotsin', 'peekaboo', 'squiddyboi']
eval_episodes = 100  # Total simulated runs per cat
max_moves = 60       # Cutoff limit from project specifications

print(f"{'Cat Name':<12} | {'Simulated':<10} | {'Failed':<8} | {'Success Rate':<12} | {'Avg Steps':<10} | {'Min Steps':<10} | {'Max Steps':<10}")
print("-" * 88)

for cat in cat_names:
    # 1. Train the bot
    q_table = train_bot(cat_name=cat, render=-1)
    
    # 2. Evaluate performance across multiple test episodes
    env = make_env(cat_type=cat)
    steps_list = []
    successes = 0
    fails = 0

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        moves = 0

        while not done and moves < max_moves:
            # Pick greedy action (pure exploitation)
            action = int(np.argmax(q_table[obs]))
            obs, _, terminated, truncated, _ = env.step(action)
            moves += 1
            done = terminated or truncated

        # Check outcome
        if done and moves <= max_moves:
            successes += 1
            steps_list.append(moves)
        else:
            fails += 1
            steps_list.append(max_moves)

    # 3. Calculate metrics
    success_rate = (successes / eval_episodes) * 100
    avg_steps = np.mean(steps_list)
    min_steps = np.min(steps_list)
    max_steps_taken = np.max(steps_list)

    print(f"{cat:<12} | {eval_episodes:>10} | {fails:>8} | {success_rate:>10.1f}% | {avg_steps:>10.2f} | {min_steps:>10} | {max_steps_taken:>10}")