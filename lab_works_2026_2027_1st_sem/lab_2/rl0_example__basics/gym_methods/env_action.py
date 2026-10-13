import gymnasium as gym


env = gym.make('CartPole-v1')
init_state, info = env.reset()
state, reward, terminated, truncated, info = env.step(0)
env.close()
