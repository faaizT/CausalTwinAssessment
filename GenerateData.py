import sys
import pandas as pd

import logging
from simulator.Model import InitialStateGenerator, physician_policy, Model


def generate_trajectory(df: pd.DataFrame, id: int):
    st = InitialStateGenerator().generate_state()
    model = Model(st)
    logging.info(f'Trajectory for Patient Id: {id}')
    for i in range(10):
        xt, ut = model.get_state().get_xt(), model.get_state().get_ut()
        action = physician_policy(xt, ut)
        row = {'id': id,
               't': i,
               'gender': xt.gender,
               'sysbp': xt.sysbp,
               'diabp': xt.diabp,
               'hr': xt.hr,
               'facial_expression': ut.facial_expression,
               'socio_econ': ut.socio_econ,
               'A_t': action}
        df = df.append(row, ignore_index=True)
        model.transition_to_next_state(action)
    return df


def main(number_of_trajectories, starting_id, export_dir):
    log_file_name = f'{export_dir}/generate-trajectories-{number_of_trajectories}-{starting_id}.log'
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    df = pd.DataFrame()
    for i in range(number_of_trajectories):
        df = generate_trajectory(df, starting_id+i)
    logging.info("Saving results to csv")
    df.to_csv(f'{export_dir}/observational-data-{number_of_trajectories}-{starting_id}.csv')


def print_help():
    print("""
    python GenerateData.py [number of trajectories] [starting id] [export-dir]
    """)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print_help()
        exit(-1)
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])