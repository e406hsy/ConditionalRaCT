# Conditional Ranking-Critical Training for Collaborative Filtering

This repository implements [Ranking-Critical Training (RaCT) for Collaborative Filtering](https://arxiv.org/abs/1906.04281) with additional information to improve quality of recommendation

This code is originally created from https://github.com/samlobel/RaCT_CF/ and modified to be able to use side information

The main result of this paper can be reproduced via running the `scripts/main_vae.ipynb` file:
## Running the code

1. Install the requirements in a **python3** environment

    `mkvirtualenv --python=python3 CRITIC_CF && pip install -r requirements.txt`

2. Install Tensorflow, version range >=1.10.0,<=1.14. 

3. Download the dataset you want to run with

    `python setup_data.py --dataset=DATASET_OF_CHOICE`
    
    
Default is *ml-100k*, which is the smallest dataset. Other options are `ml-1m`, `ml-20m`, `netflix-prize` and `msd`. Or, run with `dataset=all` to download all.
    
4. setup side information 

    `python setup_side_data.py`

This script is only applied to `ml-100k` dataset.


5. **To train a model**, run one of the scripts in the `scripts` directory named `main_*.ipynb` (see the `Main Scripts` section of the Readme). These scripts were used to generate the models and data in this paper. Or, create your own by modifying the hyperparameters or models used in one of these scripts.

6. The `train` method saves a model in a nested directory determined by its hyperparameters, within the `checkpoints` directory. It stores the data necessary for plotting in the `plotting_data` directory, and logs to the `logging` directory.
7. Passing these same hyperparameters to the `test` method as you did to `train` method will run the trained model against the held-out test data. After running `test`, results are written to the `TEST_RESULTS.txt` file, as well as to the console.

## Utils
The importable code used in the training scripts is located in the `./utils` directory.
* `warp_utils.py` contain the logic for implementing a listwise WARP-loss
* `lambdarank_utils.py` contains the logic for implementing LambdaRank
* `evaluation_functions.py` contains the logic for the ranking objective functions
* `base_models.py` and `models.py` contain the different models we train.
* `training.py` contains the `train` and `test` function used to run experiments.
* `data_loaders.py` implements importing the downloaded files as Tensorflow Datasets.

## Main Scripts

All of the scripts used to generate data are located in the `./scripts` directory.
* `main_vae.ipynb`: **This provides the paper's main result, by running the MultiVAE model with and without the critic, on all datasets.** 
* `main_vae_with_userinfo.py` is used to generate results for MultiVAE with user-category side information.
* `main_vae_with_userinfo2.py` is used to generate results for MultiVAE with user-item side information.
