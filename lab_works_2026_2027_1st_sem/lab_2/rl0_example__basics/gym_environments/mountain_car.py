from time import sleep
import gymnasium as gym


env = gym.make('MountainCar-v0', render_mode='human')

episodes = 10

for i in range(episodes):
    init_state, info = env.reset()
    reward_sum = 0
    while True:
        random_action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(random_action)
        reward_sum += reward
        sleep(0.1)
        if terminated or truncated:
            print(f'Episode_{i}, reward: {reward_sum}')
            sleep(1)
            break

env.close()
