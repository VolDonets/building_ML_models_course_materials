import random
import gymnasium as gym


env = gym.make('Blackjack-v1')
seed = 1
random.seed(seed)
init_state, info = env.reset(seed=seed)
env.close()