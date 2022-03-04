import numpy as np
import argparse
import gym
import time
import math
import random
import torch
from torch.autograd import Variable
from tqdm import tqdm
import copy
from IPython.display import clear_output
from matplotlib import image
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from utils import str2bool


class DQL:
    """ Deep Q Neural Network class. """

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.05):
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim * 2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim * 2, action_dim),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.target = copy.deepcopy(self.model)

    def update(self, state, y):
        """Update the weights of the network given a training sample. """
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        """ Compute Q values for all actions using the DQL. """
        with torch.no_grad():
            return self.model(torch.Tensor(state))

    def target_predict(self, s):
        """ Use target network to make predicitons."""
        with torch.no_grad():
            return self.target(torch.Tensor(s))

    def target_update(self):
        """ Update target network with the model weights."""
        self.target.load_state_dict(self.model.state_dict())

    def replay(self, memory, size, gamma=1.0):
        """ Add experience replay to the DQL network class."""
        if len(memory) >= size:
            # Sample experiences from the agent's memory
            data = random.sample(memory, size)
            states = []
            targets = []
            # Extract datapoints from the data
            for state, action, next_state, reward, done in data:
                states.append(state)
                q_values = self.predict(state).tolist()
                if done:
                    q_values[action] = reward
                else:
                    # The only difference between the simple replay is in this line
                    # It ensures that next q values are predicted with the target network.
                    q_values_next = self.target_predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()

                targets.append(q_values)

            self.update(states, targets)


def main(args):
    episodes = args.episodes
    run_no = args.run_num
    if args.rand_pol:
        epsilon = 1
    else:
        epsilon = args.epsilon
    env = gym.envs.make("CartPole-v1")
    model = DQL(state_dim=4, action_dim=2, hidden_dim=50, lr=0.001)
    model.model.load_state_dict(torch.load(args.pol_path))
    df = pd.DataFrame()
    for i in tqdm(range(run_no * episodes, (run_no + 1) * episodes)):
        done = False
        state = env.reset()
        t = 0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(state).argmax().item()
            df = df.append(
                {
                    "episode": i,
                    "t": t,
                    "Cart Position": state[0],
                    "Cart Velocity": state[1],
                    "Pole Angle": state[2],
                    "Pole Angular Velocity": state[3],
                    "A": action,
                },
                ignore_index=True,
            )
            state, reward, done, _ = env.step(action)
            t += 1
    if args.rand_pol:
        df.to_csv(
            f"{args.output_dir}/Cartpole-v1-sim-data-run{run_no}-ep{episodes}.csv",
            index=False,
        )
    else:
        df.to_csv(
            f"{args.output_dir}/Cartpole-v1-obs-data-run{run_no}-ep{episodes}.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        help="Number of episodes to run for",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--run_num",
        help="Run number",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--epsilon",
        help="Epsilon in eps-greedy policy",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--pol_path",
        help="Path to trained observational policy",
        type=str,
        default="/data/ziz/taufiq/export-dir/cartpole-models/obs-pol-state",
    )
    parser.add_argument(
        "--rand_pol",
        help="Generate from randomized policy",
        type=str2bool,
        default="False",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save data in",
        type=str,
        default="/data/ziz/not-backed-up/taufiq/Cartpole/simulator_data",
    )
    args = parser.parse_args()
    main(args)
