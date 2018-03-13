import gym
print(gym.__version__)

env= gym.make('CartPole-v1')


for i in range(50):
    env.render()