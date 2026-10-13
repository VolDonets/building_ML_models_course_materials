from time import sleep
import gymnasium as gym


env = gym.make('CartPole-v1', render_mode='human')
env.reset()
sleep(10)
env.close()
