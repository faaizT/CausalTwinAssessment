import argparse
import glob
import logging

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csvdir", help="path to dir containing csvs")
    parser.add_argument("outputdir", help="output file path")
    parser.add_argument("outputfilename", help="output file name")
    args = parser.parse_args()

    log_file_name = f'{args.outputdir}/combine-csv.log'
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()]
                        )
    extension = 'csv'
    all_filenames = [i for i in glob.glob(args.csvdir + '/*.{}'.format(extension))]
    logging.info("combining files:")
    logging.info(all_filenames)
    # combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    # export to csv
    logging.info("saving csv to " + args.outputdir)
    combined_csv.to_csv(args.outputdir + "/" + args.outputfilename, index=False, encoding='utf-8-sig')
    logging.info("done saving combined csv")