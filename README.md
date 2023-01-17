# CausalTwinAssessment
This code implements our methodology for Causal Assessment of Digital Twins.  

## Pulse Case Study
Below we outline steps to replicate the Pulse Case Study.

### MIMIC data extraction
First we must extract the observational dataset from [MIMIC-III database](https://mimic.mit.edu/).   

1. The MIMIC data contains detailed medical information and as such, the access to MIMIC must be requested as detailed on https://mimic.mit.edu/docs/gettingstarted/.
2. Once your application to access MIMIC has been approved, you will be granted access to the ‘MIMIC-III Clinical Database’ project page on PhysioNet:
https://physionet.org/content/mimiciii/.
3. Install MIMIC-III in a local Postgres database by following the instructions at https://mimic.mit.edu/docs/gettingstarted/local/.
Once the PSQL client is up and running you are ready to query data from MIMIC-III. 
4. Next, run the [jupyter notebook](https://github.com/matthieukomorowski/AI_Clinician/blob/master/AIClinician_Data_extract_MIMIC3_140219.ipynb) developed for the [AIClinician paper](https://www.nature.com/articles/s41591-018-0213-5) to extract MIMIC-III data. Update the `exportdir` variable in the notebook to point to the directory for saving extracted data.
5. Once the data has been extracted successfully, run the jupyter notebook [Sepsis_data_extraction](Notebooks/Sepsis_data_extraction.ipynb) to extract the data for Sepsis patients and preprocess the data. Remember to point the `exportdir` variable to the directory of previously extracted data. This notebook closely follows the preprocessing steps used in [AIClinician paper](https://www.nature.com/articles/s41591-018-0213-5) with minor modifications as outlined in our paper.  

If you successfully followed the steps outlined above, the extracted data should have been saved as `MIMICtable-1hourly.csv` in `exportdir` directory. 

### Pulse simulated data
We used the [Pulse Source Code](https://gitlab.kitware.com/physiology/engine) to obtain the simulated data for our experiments. For quick replication of our case study results, we have also included the simulated dataset in `twin_data` directory in this repository. 

If you would like to reproduce the simulated dataset from scratch, instead of using the available simulated dataset, follow these steps:  
1. Clone our fork of the Pulse Source Code from: https://gitlab.kitware.com/faaizT/engine
2. Checkout the branch `4.x`
3. Follow the instructions on https://gitlab.kitware.com/physiology/engine to build the source code 
4. Refer to the instructions at [Updating your PYTHONPATH](https://gitlab.kitware.com/physiology/engine/-/wikis/Using%20Python) to use python
5. Finally, run the file [src/python/pulse/rlengine/MIMICSimulate.py](https://gitlab.kitware.com/faaizT/engine/-/blob/4.x/src/python/pulse/rlengine/MIMICSimulate.py) from the `src/python` folder with the appropriate arguments. Specifically, `--mimicfile` should point the preprocessed MIMIC file, `--mimic_not_heldback` should point to the path to not held back mimic trajectories.

Each run of [MIMICSimulate.py](https://gitlab.kitware.com/faaizT/engine/-/blob/4.x/src/python/pulse/rlengine/MIMICSimulate.py) will generate a single trajectory. For our case study, we ran 100 simulations in parallel for a total of 48 hours to generate 26,115 twin trajectories. These are all provided in the `twin_data` directory in this repository.

### Running Hypothesis Tests
Finally, to run the hypothesis tests, simply run the following command

```python3 ./PulseHypothesisTesting.py --obs_path=<path-to-MIMIC-csv-file> --hyp_test_dir=<path-to-save-results> ```

For more details, refer to the arguments in [PulseHypothesisTesting.py](PulseHypothesisTesting.py) file.

This will save the results in the output directory.

### Visualising the results
The [Case_study_results](Notebooks/Case_study_results.ipynb) notebook includes visualisations of the saved results. Before running the code, users must change the `hyp_test_dir` variables to the folder containing the case study results.

## General Purpose Code for Causal Twin Assessment
In addition to the case study, we also provide easy to use general purpose code for implementing our methodology on different datasets. We provide a detailed tutorial in the [Hypothesis_Testing](Notebooks/Hypothesis_Testing.ipynb) notebook.