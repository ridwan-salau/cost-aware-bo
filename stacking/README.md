# How to Run The Stacking Experiment

Create a new environment and install the `requirements.txt` file as follows:

```{console}
conda create -n stackingenv
conda activate stackingenv
pip install -r stacking/requirements.txt
```

> Install the cost_aware_bo package from the root directory. The -e flag installs it in editable mode such that when you make changes to the package, you don't have to re-install it.

```{console}
pip install -e .
```

Download the stacking dataset from [here](https://filebin.net/ikioyab5pg2mehtl/stacking_dataset.zip). Unzip the file and place the files inside `stacking/inputs`.

Run the following command from inside the `stacking/` directory to start the stacking experiment:

```{console}
bash run_stacking.sh
```

If you wish to disable WANDB logging, run the following:

```{console}
WANDB_MODE=offline bash run_stacking.sh
```