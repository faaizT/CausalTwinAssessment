import argparse
from gumbel_max_sim.SimulatorDataGenerator import SimulatorDataGenerator
import logging

def main(args):
    data_generator = SimulatorDataGenerator(args.model_path, args.simulator)
    data_generator.save_trajectories(args.n_trajectories, args.exportdir)
    logging.info("Generated data")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("exportdir", help="path to output directory")
    parser.add_argument("--n_trajectories", help="number of trajectories", type=int, default=10000)
    parser.add_argument("--simulator", help="name of simulator to run", type=str, default='real')
    parser.add_argument("--model_path", help="path to saved model", type=str)
    args = parser.parse_args()
    main(args)
