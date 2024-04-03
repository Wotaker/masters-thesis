# masters-thesis
This is a repository to support my master's thesis research. The topic of the thesis is *Analysis of Brain Effective Connectivity with Reservoir Computing Approach*.

## Repository structure

The main repository directory is `master_thesis` which containes all the source code to process the connectivity data. It is organized as follows:

- **`classification_models`:** Contains the implemented ML models to classify the connectivity graphs. The models API is a standard scikit-learn API, with a `fit` and `predict` methods. When a model is trained, its checkpoints are saved in the `classification_models/checkpoints/$MODEL_ALIAS` directory.  

- **`training`:** Contains the main training script to train the models. The script `training/train.py` defines both holdout and cross-validation training procedures. Training configurations, hyperparameters, preprocessing steps, and dataset paths are defined in config `.yml` files located in the `training/configs` directory.  
  
- **`tools`:** This directory contains utility functions to load and preprocess the data, located in the `tools/data` directory, plotting utilities in the `tools/plotting` directory, and a set python scripts to organize files, generate synthetic data and summarize results in the `tools/scripts` directory.

### Other important directories

When reproducing the results, the following directories are important:

- **Networks Directory** The directory which contains the connectivity networks in the form of .npy files.
- **Results Directory** The directory where the results of the experiments will be saved.

## Datasets

The dataset directory is not included in this repository, although the data can be shared on request. The data can be organized in any way but the important thing is the connectivity networks are stored in a designated directory with per subject .npy files, which shapes are either `(n_channels, n_nodes, n_nodes)` or just `(n_nodes, n_nodes)`. These networks can be obtained generally in any way, however in this thesis I focus on the whole-brain Reservoir Computing Causality (whole-brain RCC) aproach and the Granger Causality (GC) approach. These were generated using the *Effective-Connectivity-Reservoir-Computing* [repository](https://github.com/JoanSano/Effective-Connectivity-Reservoir-Computing/tree/master).

## MLOps

To keep track of the experiments I've defined a set of procedures. First, the experiment configuration are defined in `.yml` config files located in the `training/configs` directory. Then, the `training/train.py` accepts as the argument the path to this config file together with a path to a parent results directory where the experiments results will be stored in a seperate directory starting with a timestamp. When the experiment is run, a few things are saved:
- the confusion matrices from classification of both train and test sets,
- the `results.csv` file with the classification metrics from each model, dataset and fold,
- and finally the config file used to run the experiment is copied to this results directory.

Than, when the experiment is finished, one can process the results to verify the hypotheses stated by the tesis.
- The `tools/scripts/test_hypothesis_I.py` script summarizes the `results.csv` to aggregate the classification metrics and present them in a more readable way.
- The `tools/scripts/test_hypothesis_II.py` is to determin wether one approach of computing EC is better than the other. It accepts name of the assumed superior and inferior dataset (containing networks with computed EC) and tests the one-sided t-test that the superior dataset has a higher AUC than the inferior dataset.

## Reproducing the results

To reproduce the results, one should follow the steps below:

1. Prepare the dataset directory with the connectivity networks, lets name it `NETWORKS_DIR`.
2. Define a training configuration in a `.yml` file located in the `training/configs` directory. One can use the existing `training/configs/sample_config.yml` as a template.
3. `cd` to the repository root directory.
4. Setup the python package and install the dependencies by running the following command (I strongly recommend using a virtual environment with python 3.10):  
   `pip install -e .`
5. Run the training script:  
   `python master_thesis/training/train.py -c master_thesis/training/configs/sample_config.yml`
6. After the training is finished, the results are saved in the default `Results` directory. One can change the default results directory by passing the `-r` argument to the training script.
7. (Optional) To summarize the results, and test the hypotheses, run the scripts in the `tools/scripts` directory:
   1. `python master_thesis/tools/scripts/test_hypothesis_I.py -r $EXPERIMENT_RESULTS_DIR`
   2. `python master_thesis/tools/scripts/test_hypothesis_II.py -r $EXPERIMENT_RESULTS_DIR -s $SUPERIOR_DATASET -i $INFERIOR_DATASET`

