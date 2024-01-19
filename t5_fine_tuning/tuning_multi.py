import json
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Tuple, Union

import datasets
import torch
from cachestore import Cache, LocalStorage
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from utils import (
    distillation,
    inference,
    load_hyperparameters,
    select_first_n_stages,
    tuning,
)

parser = ArgumentParser()
parser.add_argument(
    "--exp-name",
    type=str,
    help="Specifies a unique experiment name",
)
parser.add_argument("--trial", type=int, help="The trial number", default=1)
parser.add_argument(
    "--cache-root", type=Path, default=".cachestore", help="Cache directory"
)
parser.add_argument(
    "--acqf",
    type=str,
    help="Acquisition function",
    choices=["EEIPU", "MS_CArBO", "EIPS", "CArBO", "EI", "RAND"],
    default="EI",
)

args, _ = parser.parse_known_args()

disable_cache = args.acqf != "EEIPU"
cache = Cache(
    f"stacking_{args.exp_name}_{args.trial}_cache",
    storage=LocalStorage(args.cache_root),
    disable=disable_cache,
)

num_samples = 2500  # 9540 seconds to finish 25K samples on 4 gpus


# @task(cache=True, cache_key_file="hparams", timer=True)
@cache(ignore={"output_dir", "dataset"})
def data_preprocessing(
    dataset: Union[Path, str], output_dir: Union[Path, str], *, hparams
):
    """Data Preprocessing Stage.

    dataset (Path or str): path where the training dataset and the hyperparameters dataset are stored.
    output_dir (Path or str): path where output results are stored. Must be unique for each run
    """
    dataset = Path(dataset)
    output_dir = Path(output_dir)

    # start_time = time.time()
    train_inputs_encodings, train_summaries_encodings = torch.load(
        dataset / "tokenized_train_data.pt"
    )
    val_inputs_encodings, val_summaries_encodings = torch.load(
        dataset / "tokenized_validation_data.pt"
    )

    if num_samples > 0:
        train_inputs_encodings = {
            key: value[:num_samples] for key, value in train_inputs_encodings.items()
        }
        train_summaries_encodings = {
            key: value[:num_samples] for key, value in train_summaries_encodings.items()
        }
        val_inputs_encodings = {
            key: value[:num_samples] for key, value in val_inputs_encodings.items()
        }
        val_summaries_encodings = {
            key: value[:num_samples] for key, value in val_summaries_encodings.items()
        }

    train_dataset = TensorDataset(
        train_inputs_encodings["input_ids"],
        train_inputs_encodings["attention_mask"],
        train_summaries_encodings["input_ids"],
        train_summaries_encodings["attention_mask"],
    )

    val_dataset = TensorDataset(
        val_inputs_encodings["input_ids"],
        val_inputs_encodings["attention_mask"],
        val_summaries_encodings["input_ids"],
        val_summaries_encodings["attention_mask"],
    )

    batch_size = hparams["0__batch_size"] * torch.cuda.device_count()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
    )

    original_data = load_dataset("cnn_dailymail", "3.0.0")
    test_dataset = original_data["test"]
    test_dataset = test_dataset.shuffle(seed=42).select([i for i in range(10)])

    # Write the data in the output directory.
    test_dataset.save_to_disk(output_dir / "testing_data")
    torch.save(train_dataloader, output_dir / "training_data")
    torch.save(val_dataloader, output_dir / "validation_data")
    with (output_dir / "data_preprocessing.txt").open("w") as f:
        f.write("Data preprocessing has been completed!")

    # end_time = time.time()
    # metrics = {"cost": end_time - start_time}
    # with open(output_dir / "metrics.json", "w") as metrics_file:
    #     json.dump(metrics, metrics_file)

    return output_dir


# @task(cache=True, cache_key_file="hparams", timer=True)
@cache(ignore={"output_dir", "dataset", "data_prepoc_output_path", "epochs"})
def fine_tuning(
    dataset: Union[Path, str],
    data_prepoc_output_path: Union[Path, str],
    output_dir: Union[Path, str],
    epochs,
    *,
    hparams,
    global_epochs,
    model_name: Union[Path, str],
) -> Path:
    """Fine Tuning Stage."""
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    data_prepoc_output_path = Path(data_prepoc_output_path)

    # start_time = time.time()
    model_path = tokenizer_path = model_name
    if isinstance(model_name, Path) and model_name.exists():
        model_path = model_path / "fine_tuned_model"
        tokenizer_path = tokenizer_path / "fine_tuned_tokenizer"

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda:0")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["1__learning_rate"],
        weight_decay=hparams["1__weight_decay"],
    )

    train_dataloader = torch.load(data_prepoc_output_path / "training_data")
    val_dataloader = torch.load(data_prepoc_output_path / "validation_data")
    validation_dataset = datasets.load_from_disk(dataset / "validation_data")
    if num_samples > 0 and num_samples < 13368:
        validation_dataset = validation_dataset.select(range(num_samples))
    metrics, fine_tuned_model, fine_tuned_tokenizer = tuning(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        # num_epochs__1,
        # hparams["1__num_epochs"],
        epochs,
        tokenizer,
        validation_dataset,
    )

    model_chkpt_path = output_dir / str(global_epochs)
    fine_tuned_model.save_pretrained(model_chkpt_path / "fine_tuned_model/")
    fine_tuned_tokenizer.save_pretrained(model_chkpt_path / "fine_tuned_tokenizer/")

    # metrics["epoch_num"].append(hparams["1__num_epochs"])
    metrics["learning_rate"].append(hparams["1__learning_rate"])
    metrics["batch_size"].append(hparams["0__batch_size"])

    # end_time = time.time()
    # metrics["cost"] = end_time - start_time

    # with open(output_dir / "metrics.json", "w") as metrics_file:
    #     json.dump(metrics, metrics_file)

    print(f"Epoch {global_epochs} fine tuning completed")

    return model_chkpt_path


# @task(cache=True, cache_key_file="hparams")
@cache(
    ignore={
        "output_dir",
        "dataset",
        "fine_tuned_model_path",
        "data_prepoc_output_path",
        "epochs",
    }
)
def model_distillation(
    dataset: Union[Path, str],
    data_prepoc_output_path: Union[Path, str],
    fine_tuned_model_path: Union[Path, str],
    output_dir: Union[Path, str],
    epochs,
    *,
    hparams,
    student_model_name: Union[Path, str],
    global_epochs,
) -> Tuple[float, Path]:
    """Distillation Stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # start_time = time.time()
    # model_name = "t5-small"

    # Load your fine-tuned t5-small teacher model and tokenizer
    teacher_model = T5ForConditionalGeneration.from_pretrained(
        fine_tuned_model_path / "fine_tuned_model/"
    ).to("cuda")

    student_model_path = student_tokenizer_path = student_model_name
    if isinstance(student_model_name, Path) and student_model_name.exists():
        student_model_path = student_model_path / "distilled_model"
        student_tokenizer_path = student_tokenizer_path / "distilled_tokenizer"

    # Load T5-tiny student model
    tokenizer = T5Tokenizer.from_pretrained(
        student_tokenizer_path, d_model=128, d_ff=512, d_kv=64, num_layers=2
    )
    student_config = T5Config.from_pretrained(
        student_model_path, d_model=128, d_ff=512, d_kv=64, num_layers=2
    )
    student_model = T5ForConditionalGeneration(student_config).to("cuda")

    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=hparams["2__learning_rate"],
        weight_decay=hparams["2__weight_decay"],
    )

    # Define your training & validation dataset and dataloader
    train_dataloader = torch.load(data_prepoc_output_path / "training_data")
    val_dataloader = torch.load(data_prepoc_output_path / "validation_data")
    validation_dataset = datasets.load_from_disk(dataset / "validation_data")

    if num_samples > 0 and num_samples < 13368:
        validation_dataset = validation_dataset.select(range(num_samples))

    metrics, distilled_model, distilled_tokenizer = distillation(
        teacher_model,
        student_model,
        train_dataloader,
        val_dataloader,
        optimizer,
        # num_epochs__2,
        # hparams["2__num_epochs"],
        epochs,
        tokenizer,
        validation_dataset,
        hparams["2__temperature"],
        output_dir,
    )

    model_chkpt_path = output_dir / str(global_epochs)
    distilled_model.save_pretrained(model_chkpt_path / "distilled_model")
    distilled_tokenizer.save_pretrained(model_chkpt_path / "distilled_tokenizer")

    # metrics["epoch_num"].append(hparams["2__num_epochs"])
    metrics["learning_rate"].append(hparams["2__learning_rate"])
    metrics["batch_size"].append(hparams["0__batch_size"])
    # end_time = time.time()
    # metrics["cost"] = end_time - start_time
    # with open(output_dir / "metrics.json", "w") as metrics_file:
    #     json.dump(metrics, metrics_file)

    print(f"Epoch {global_epochs} of distillation completed")

    return metrics["rouge_scores"][0]["rougeLsum"], model_chkpt_path


# @task(cache=True, cache_key_file="hparams")
def model_inference(
    data_prepoc_output_path: Union[Path, str],
    fine_tuned_model_path: Union[Path, str],
    # dataset: Union[Path, str],
    distilled_model_path: Union[Path, str],
    output_dir: Union[Path, str],
    *,
    hparams,
):
    start_time = time.time()
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained(
        fine_tuned_model_path / "fine_tuned_model/"
    )
    fine_tuned_tokenizer = T5Tokenizer.from_pretrained(
        fine_tuned_model_path / "fine_tuned_tokenizer/"
    )
    # Load the distilled model
    distilled_model = T5ForConditionalGeneration.from_pretrained(
        distilled_model_path / "distilled_model/"
    )
    # Quantize the distilled model
    quantized_distilled_model = torch.quantization.quantize_dynamic(
        distilled_model, {torch.nn.Linear}, dtype=torch.qint8
    )
    quantized_distilled_tokenizer = T5Tokenizer.from_pretrained(
        distilled_model_path / "distilled_tokenizer/"
    )
    test_dataset = datasets.load_from_disk(data_prepoc_output_path / "testing_data")
    test_data = test_dataset["article"]
    reference_summaries = test_dataset["highlights"]

    # Inference for fine-tuned model
    fine_tuned_metrics = inference(
        fine_tuned_model, fine_tuned_tokenizer, test_data, reference_summaries
    )

    # Inference for quantized model
    quantized_metrics = inference(
        quantized_distilled_model,
        quantized_distilled_tokenizer,
        test_data,
        reference_summaries,
    )

    metrics = {
        "fine_tuned_metrics": fine_tuned_metrics,
        "quantized_metrics": quantized_metrics,
        "cost": 0,
    }
    end_time = time.time()
    metrics["cost"] = end_time - start_time

    with open(output_dir / "metrics.json", "w") as metrics_file:
        json.dump(metrics, metrics_file)

    print("Inference has been completed")

    return output_dir


# @task(cache=True, cache_key_file="hparams")
def generate_output(
    data_prepoc_output_path: Union[Path, str],
    fine_tuned_model_path: Union[Path, str],
    distilled_model_path: Union[Path, str],
):
    """Output Generation."""
    # with open(dataset / "initial_hparams.json") as initial_hparams:
    #     initial_hps = json.load(initial_hparams)

    final_metrics = {}

    # with open(dataset / "hparams.json") as _hparams:
    #     hyperparmeters = json.load(_hparams)

    # hp_dataset = hyperparmeters["dataset"]

    with open(data_prepoc_output_path / "metrics.json") as _metrics:
        data_metrics = json.load(_metrics)

    with open(fine_tuned_model_path / "metrics.json") as _metrics:
        fine_tuning_metrics = json.load(_metrics)

    with open(distilled_model_path / "metrics.json") as _metrics:
        distillation_metrics = json.load(_metrics)

    # with open(inference_output / "metrics.json") as _metrics:
    #     inference_metrics = json.load(_metrics)

    final_metrics["cost"] = [
        data_metrics["cost"],
        fine_tuning_metrics["cost"],
        distillation_metrics["cost"],
        # inference_metrics["cost"],
    ]

    final_metrics["obj"] = distillation_metrics["rouge_scores"][0]["rougeLsum"]

    # obj_validation_loss = metrics["validation_loss"]
    # obj_validation_loss_avg = sum(obj_validation_loss) / len(obj_validation_loss)

    # hp_dataset["obj"].append(obj_validation_loss_avg)

    # initial_hps["dataset"] = hp_dataset

    # with (output_dir / "hparams_file.json").open("w") as f:
    #     json.dump(final_metrics, f)

    return final_metrics


def t5_fine_tuning(
    dataset: Union[Path, str],
    output_dir: Union[Path, str],
    stg_hparams: List[Dict],
    ft_num_epochs: int = 5,
    fine_tune_num_stgs: int = 1,
    dstl_num_epochs: int = 15,
    dstl_num_stgs: int = 3,
):
    """Main Pipeline."""
    print(f"HPs for this iteration:\n\n{stg_hparams}\n\n")
    tot_num_stgs = 1 + fine_tune_num_stgs + dstl_num_stgs
    tot_num_hp_stgs = len(set(int(key.split("__")[0]) for key in stg_hparams))
    assert (
        tot_num_stgs == tot_num_hp_stgs
    ), f"The total number of stages in the pipeline, {tot_num_stgs}, is not equal to the number hyperparameter stages {tot_num_hp_stgs}"

    curr_stg = 1
    data_preproc_hp = select_first_n_stages(stg_hparams, curr_stg)
    tuning_hps = [
        select_first_n_stages(stg_hparams, stg + 1 + curr_stg)
        for stg in range(fine_tune_num_stgs)
    ]
    curr_stg += fine_tune_num_stgs
    distillation_hps = [
        select_first_n_stages(stg_hparams, stg + 1 + curr_stg)
        for stg in range(dstl_num_stgs)
    ]

    # Stage 1: Data preprocessing
    all_stages_costs = []
    start_data_proc = time.time()
    data_prepoc_output_path = data_preprocessing(
        dataset, output_dir / "data_preprocessing", hparams=data_preproc_hp
    )

    # Stage 2: Fine-tuning
    start_fine_tune = time.time()
    all_stages_costs.append(start_fine_tune - start_data_proc)
    fine_tuned_model_path = "t5-small"
    global_epochs = 0
    ft_epochs_per_stage = (ft_num_epochs // fine_tune_num_stgs) + (
        1 if (ft_num_epochs % fine_tune_num_stgs) else 0
    )
    while global_epochs < ft_num_epochs:
        epochs = min(ft_epochs_per_stage, ft_num_epochs - global_epochs)
        print(epochs)
        fine_tuned_model_path = fine_tuning(
            dataset,
            data_prepoc_output_path,
            output_dir / "fine_tuning",
            epochs,
            hparams=tuning_hps.pop(0),
            global_epochs=global_epochs,
            model_name=fine_tuned_model_path,
        )
        global_epochs += ft_epochs_per_stage

        all_stages_costs.append(time.time() - start_fine_tune)
        start_fine_tune = time.time()

    # Stage 3: Distillation
    start_distil = time.time()
    distilled_model_path = "t5-small"
    global_epochs = 0
    dstl_epochs_per_stage = (dstl_num_epochs // dstl_num_stgs) + (
        1 if (dstl_num_epochs % dstl_num_stgs) else 0
    )
    while global_epochs < dstl_num_epochs:
        epochs = min(dstl_epochs_per_stage, dstl_num_epochs - global_epochs)
        print(epochs)
        rougeLsum, distilled_model_path = model_distillation(
            dataset,
            data_prepoc_output_path,
            fine_tuned_model_path,
            output_dir / "model_distillation",
            epochs,
            hparams=distillation_hps.pop(0),
            student_model_name=distilled_model_path,
            global_epochs=global_epochs,
        )
        global_epochs += dstl_epochs_per_stage

        all_stages_costs.append(time.time() - start_distil)
        start_distil = time.time()

    return {"obj": rougeLsum, "costs": all_stages_costs}


if __name__ == "__main__":
    start = time.time()
    dataset = Path("inputs")
    output_dir = Path("outputs") / time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    stg_hparams = load_hyperparameters(dataset / "hparams_multi.json")
    print(stg_hparams)
    output = t5_fine_tuning(
        dataset,
        output_dir,
        stg_hparams,
        ft_num_epochs=2,
        fine_tune_num_stgs=1,
        dstl_num_epochs=6,
        dstl_num_stgs=3,
    )
    print(output)
    print("Total duration:", time.time() - start)
