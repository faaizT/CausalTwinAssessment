import argparse
import sys
import pandas as pd

import logging
from observational_model.Model import InitialStateGenerator, physician_policy, Model
from simulator.Simulator import RealSimulator, get_simulator


def generate_trajectory(model_name: str, df: pd.DataFrame, trajectory_length: int, id: int):
    model = get_simulator(model_name)
    logging.info(f'Trajectory for Patient Id: {id}')
    for i in range(trajectory_length):
        xt, ut = model.get_state().get_xt(), model.get_state().get_ut()
        action = physician_policy(xt, ut)
        row = {'id': id, 't': i, 'A_t': action}
        row.update(xt.as_dict())
        row.update(ut.as_dict())
        df = df.append(row, ignore_index=True)
        model.transition_to_next_state(action)
    return df


def main(model_name, number_of_trajectories, trajectory_length, starting_id, export_dir):
    log_file_name = f'{export_dir}/generate-trajectories-{number_of_trajectories}-{starting_id}.log'
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    df = pd.DataFrame()
    for i in range(number_of_trajectories):
        df = generate_trajectory(model_name, df, trajectory_length, starting_id+i)
    logging.info("Saving results to csv")
    df.to_csv(f'{export_dir}/observational-data-{number_of_trajectories}-{starting_id}.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectories", help="number of trajectories to be generated", type=int)
    parser.add_argument("trajectory_len", help="length of each trajectory", type=int)
    parser.add_argument("starting_id", help="starting id", type=int)
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--name", help="model name to be used", default="real")

    args = parser.parse_args()
    main(model_name=args.name, number_of_trajectories=args.trajectories, trajectory_length=args.trajectory_len,
         starting_id=args.starting_id, export_dir=args.exportdir)
