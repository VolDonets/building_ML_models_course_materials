from time import sleep
import gymnasium as gym


env = gym.make('Phoenix-v0', render_mode='human')
print(f'Action Space: {env.action_space}')
print(f'Observation Space: {env.observation_space}')

episodes = 10

for i in range(episodes):
    init_state, info = env.reset()
    reward_sum = 0
    while True:
        random_action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(random_action)
        reward_sum += reward
        sleep(.01)
        if terminated or truncated:
            print(f'Episode {i} reward: {reward_sum}')
            sleep(1)
            break

env.close()