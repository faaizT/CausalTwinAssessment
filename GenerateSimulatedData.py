import sys
import pandas as pd
import logging
import ntpath

from observational_model.PatientState import Xt
from simulator.Simulator import get_simulator


def simulate_trajectories(obs_data_file, simulator_name, samples_per_obs, export_dir, epsilon=5.0):
    obs_data_file_name = ntpath.basename(obs_data_file)
    log_file_name = f'{export_dir}/simulated-trajectories-{simulator_name}-{obs_data_file_name}.log'
    logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
    obs_data = pd.read_csv(obs_data_file)
    ids = set(obs_data['id'])
    logging.info(f"Found {len(ids)} ids.")
    results_df = pd.DataFrame()
    for id in ids:
        obs_data_filtered = obs_data[obs_data['id'] == id].sort_values(by=['t'], inplace=False)
        j = 0
        while j < samples_per_obs:
            logging.info(f"On id: {id} --------- Finding sample no: {j}")
            trajectory = []
            simulator = get_simulator(simulator_name)
            st = simulator.get_state()
            xt_sim = st.get_xt()
            for index, row in obs_data_filtered.iterrows():
                if Xt.from_series(row).distance(xt_sim) <= epsilon:
                    trajectory.append((st, xt_sim, int(row['A_t'])))
                    if index == len(obs_data_filtered)-1:
                        j += 1
                        if len(trajectory) != len(obs_data_filtered):
                            raise Exception("Trajectory of unexpected length")
                        for i in range(len(trajectory)):
                            logging.info("Found a trajectory!")
                            data = {'id': id, 't': i, 'A_t': trajectory[i][2]}
                            data.update(trajectory[i][0].as_dict())
                            data.update(trajectory[i][1].as_dict())
                            results_df = results_df.append(data, ignore_index=True)
                    else:
                        simulator.transition_to_next_state(int(row['A_t']))
                        st = simulator.get_state()
                        xt_sim = st.get_xt()
                else:
                    break
        logging.info("Updating csv file")
        results_df.to_csv(f"{export_dir}/simulated-data-{simulator_name}-{obs_data_file_name}.csv", index=False)


def print_help():
    print("""
    python GenerateSimulatedData.py [obs data file-path] [simulator name] [samples per obs] [export_dir] [(optional) epsilon]
    """)


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print_help()
        exit(-1)
    if len(sys.argv) == 5:
        simulate_trajectories(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        simulate_trajectories(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], float(sys.argv[5]))
