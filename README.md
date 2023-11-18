# COST-AWARE BAYESIAN HYPERPARAMETER OPTIMIZATION OF ML PIPELINES WITH BLACKBOX COST

## Steps to run the T5 Pipeline Tuning Experiment
### Option A
- Clone the repository `git clone https://github.com/ridwan-salau/cost-aware-bo.git` checkout to ridwan/t5-pipeline `git checkout ridwan/t5-pipeline`
- Download the datasets - [tokenized_train_data.pt](https://github.com/maazmaqsood/pirlib/blob/fine-tuning-pipeline/examples/t5_fine_tuning/inputs/tokenized_train_data.pt) and [tokenized_validation_data.pt](https://github.com/maazmaqsood/pirlib/blob/fine-tuning-pipeline/examples/t5_fine_tuning/inputs/tokenized_validation_data.pt). (N.B. you might need git-lfs to be able to pull the file from github as downloading larges files from git directly won't work.)
- Place the two files in the directory `t5_fine_tuning/inputs/`
- Create a conda environment by running `conda create -f t5_fine_tuning/t5env.yml` from the root.
- To setup WandB logging, run `export WANDB_API_KEY=<<API KEY FROM WANDB>>`
- Run an experiment using `bash run.sh <<ACQF_Method>>`, where `ACQF_Method` is one of `{EI, EEIPU, EIPS, CArBO}`

### Option B (Docker)
- Clone the repository `git clone https://github.com/ridwan-salau/cost-aware-bo.git` checkout to ridwan/t5-pipeline `git checkout ridwan/t5-pipeline`
- Place 