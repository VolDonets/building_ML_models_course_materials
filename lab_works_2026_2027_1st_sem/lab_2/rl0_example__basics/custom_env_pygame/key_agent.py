import random
from time import sleep
from catch_coins_env import CatchCoinsEnv

# 1. Pass render_mode during initialization, not during the render() call
env = CatchCoinsEnv(render_mode='ascii')

# 2. reset() now returns two values: observation and info
init_state, info = env.reset()

# -1: left
#  0: stay
# +1: right
action_space = [-1, 0, 1]

for _ in range(1000):
    # 3. render() no longer takes the mode argument
    env.render()

    # Added a small hint to the input prompt for clarity (1=left, 2=stay, 3=right)
    print("push step (1, 2, or 3): ")
    action = int(input())
    action -= 2

    # 4. step() now returns 5 values instead of 4
    state, reward, terminated, truncated, info = env.step(action)

    # Standard practice: check if the environment is done (even if your custom env runs forever)
    done = terminated or truncated
    if done:
        break

    sleep(.2)

env.close()
