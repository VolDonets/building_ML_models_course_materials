from time import sleep
import gymnasium as gym
import random

# init env with render_mode for visualization
env = gym.make('CartPole-v1', render_mode='human')

# making the script reproducible
seed = 0
random.seed(seed)

print(f'Action Space: {env.action_space}\n')
print(f'Observation Space: {env.observation_space}')

# run 10 episodes in a row
episodes = 10
for i in range(episodes):
    # reset returns a tuple: (observation, info)
    init_state, info = env.reset(seed=seed if i == 0 else None)
    reward_sum = 0

    while True:
        # Generate a random action (0 or 1)
        # Note: You can also use `env.action_space.sample()` instead of random.randint
        random_action = random.randint(0, 1)

        # step() returns 5 values instead of 4
        state, reward, terminated, truncated, info = env.step(random_action)

        done = terminated or truncated
        reward_sum += reward
        sleep(0.1)

        if done:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(0.1)
            break

env.close()
