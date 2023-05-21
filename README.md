This code was used for all the analyses in PAPER!!!. If you use it to publish any results, please cite (!!!).

# Repo Summary

## Two-Layer-Linear Network Analyses (2LL - Section 3)

* `2LL_results/`: folder for storing results of model fitting obtained from running `2LL_sim_script.py`
* `2LL_summaries/`: folder for storing summarized results obtained from running `2LL_summary_script.py`
* `2LL_sim_script.py`: script used to fit the model for a single $Q,Q^*$ pair and a range of $\beta$, to many instances of data generated from a single data-generator.
* `2LL_summary_script.py`: summarizes results of model fitting across different data-generators into a single file in `2LL_summaries/`

## Convolutional Neural Network on Digits Dataset Analyses (digits - Section 4)

* `digits_results/`: folder for storing results of model fitting obtained from running `digits_sim_script.py`
* `digits_summaries/`: folder for storing summarized results obtained from running `digits_summary_script.py` or `digits_subsample_sum_script.py`
* `digits_sim_script.py`: methods used to train a data-generating CNN and fit a joint-trained model for a single choice of hyperparameters.
* `digits_run_main.py`: script executed to fit models for a range of true and hypothesized computations.
* `digits_run_subsample.py`: script executed to fit models for a range of degrees of subsampling.
* `digits_summary_script.py`: summarizes results of model fitting across all different true and hypothesized computations into files in `digits_summaries/`
* `digits_subsample_sum_script.py`: summarizes results of model fitting across different degrees of subsampling into a single file in `digits_summaries/`

## Shared
* `readme.md`: this file, a guide to this repo
* `figs/`: folder for saved figures
* `Figures.ipynb`: notebook for generating figures
* `slurm_script.sh`: script for submitting jobs on a SLURM computing cluster


# Result Reproduction

## Two-Layer-Linear Network Analyses (2LL - Section 3)
1. Edit `slurm_script.sh` to run `digits_sim_script.py` with array_ids 0-179 then run `sbatch slurm_script.sh`
2. When it is finished running, run `python3 2LL_summary_script.py`
3. You can now run the 2LL cells in Figures.ipynb

## Convolutional Neural Network on Digits Dataset Analyses (digits - Section 4)
1. Edit `slurm_script.sh` to run `digits_run_main.py` with array_ids 0-103679 then run `sbatch slurm_script.sh`
2. When it is finished running, run `python3 digits_summary_script.py`
3. You can now run the digits cells in Figures.ipynb, except the subsampling one
4. Edit `slurm_script.sh` to run `digits_run_subsample.py` with array_ids 0-25919 then run `sbatch slurm_script.sh`
5. When it is finished running, run `python3 digits_subsample_sum_script.py`
6. You can now run all digits cells in Figures.ipynb




