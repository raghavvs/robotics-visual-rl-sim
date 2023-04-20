import gym
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

env = gym.make('FetchReach-v1')

model = DQN(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=100000)

for episode in range(10):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()