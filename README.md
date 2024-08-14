# COST-AWARE BAYESIAN HYPERPARAMETER OPTIMIZATION OF ML PIPELINES WITH BLACKBOX COST

## Steps to run the T5 Pipeline Tuning Experiment
- Clone the repository `git clone https://github.com/ridwan-salau/cost-aware-bo.git` checkout to ridwan/t5-pipeline `git checkout ridwan/t5-pipeline`
- Download the datasets - [tokenized_train_data.pt](https://github.com/maazmaqsood/pirlib/blob/fine-tuning-pipeline/examples/t5_fine_tuning/inputs/tokenized_train_data.pt) and [tokenized_validation_data.pt](https://github.com/maazmaqsood/pirlib/blob/fine-tuning-pipeline/examples/t5_fine_tuning/inputs/tokenized_validation_data.pt). (N.B. you might need git-lfs to be able to pull the file from github as downloading larges files from git directly won't work.)
- Place the two files in the directory `t5_fine_tuning/inputs/`
- Create a python environment using `pip` or `conda`, and install `t5_fine_tuning/requirements.txt`.
- To setup WandB logging, run `export WANDB_API_KEY=<<API KEY FROM WANDB>>` or turn off logging with `export WANDB_MODE=offline`.
- Run an experiment from the `t5_fine_tuning/` directory using `bash run_multi.sh`.

## Steps to run the T5 Pipeline Tuning Experiment On Slurm
- Clone the repository `git clone https://github.com/ridwan-salau/cost-aware-bo.git` checkout to ridwan/t5-pipeline `git checkout ridwan/t5-pipeline-slurm`
- Download the datasets - [tokenized_train_data.pt](https://github.com/maazmaqsood/pirlib/blob/fine-tuning-pipeline/examples/t5_fine_tuning/inputs/tokenized_train_data.pt) and [tokenized_validation_data.pt](https://github.com/maazmaqsood/pirlib/blob/fine-tuning-pipeline/examples/t5_fine_tuning/inputs/tokenized_validation_data.pt). (N.B. you might need git-lfs to be able to pull the file from github as downloading larges files from git directly won't work.)
- Place the two files in the directory `t5_fine_tuning-slurm/inputs/`
- Create a python environment using `pip` or `conda`, and install `t5_fine_tuning-slurm/requirements.txt`.
- To setup WandB logging, run `export WANDB_API_KEY=<<API KEY FROM WANDB>>` or turn off logging with `export WANDB_MODE=offline`.
- Run an experiment from the `t5_fine_tuning-slurm/` directory using

```
sbatch submit_batch.sh ../t5_fine_tuning/inputs t5-small
```

where `t5_fine_tuning/inputs` is the inputs directory and `t5-small` is the T5 base model to be finetuned.

## Steps to run the Stacking Pipeline Experiment
- [Stacking Doc](./stacking/README.md)