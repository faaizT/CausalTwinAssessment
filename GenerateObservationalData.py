import sys
import pandas as pd

import logging
from observational_model.Model import InitialStateGenerator, physician_policy, Model
from simulator.Simulator import RealSimulator


def generate_trajectory(df: pd.DataFrame, trajectory_length: int, id: int):
    st = InitialStateGenerator().generate_state()
    model = RealSimulator(st)
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


def main(number_of_trajectories, trajectory_length, starting_id, export_dir):
    log_file_name = f'{export_dir}/generate-trajectories-{number_of_trajectories}-{starting_id}.log'
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    df = pd.DataFrame()
    for i in range(number_of_trajectories):
        df = generate_trajectory(df, trajectory_length, starting_id+i)
    logging.info("Saving results to csv")
    df.to_csv(f'{export_dir}/observational-data-{number_of_trajectories}-{starting_id}.csv', index=False)


def print_help():
    print("""
    python GenerateObservationalData.py [number of trajectories] [length of trajectories] [starting id] [export-dir]
    """)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print_help()
        exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
