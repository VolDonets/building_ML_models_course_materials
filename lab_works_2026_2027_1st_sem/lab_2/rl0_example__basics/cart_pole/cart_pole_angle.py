from time import sleep
import gymnasium as gym
import random

# init env with render_mode for visualization
env = gym.make('CartPole-v1', render_mode='human')

# making the script reproducible
seed = 0
random.seed(seed)
# Note: env.seed() is deprecated. Seeding is now handled in env.reset()

print(f'Action Space: {env.action_space}\n')
print(f'Observation Space: {env.observation_space}')

# run 10 episodes in a row
episodes = 10
for i in range(episodes):
    # reset returns a tuple: (observation, info).
    # We apply the seed only on the first episode to ensure the sequence matches.
    state, info = env.reset(seed=seed if i == 0 else None)
    reward_sum = 0

    while True:
        # env.render() is no longer needed in the loop when render_mode='human' is set during make()
        action = 1 if state[2] > 0 else 0

        # step() now returns 5 values: observation, reward, terminated, truncated, info
        state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        reward_sum += reward
        sleep(.01)

        if done:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(1)
            break

env.close()
