import gymnasium as gym


env = gym.make('Blackjack-v1')
print(env.action_space)
env.close()
