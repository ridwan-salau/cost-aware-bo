import json
import time
import math
from copy import deepcopy
from argparse import ArgumentParser

from pathlib import Path
from typing import Union, List, Dict, Any
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from utils import tuning, distillation, load_hyperparameters, inference

from cachestore import Cache, LocalStorage

parser = ArgumentParser()
parser.add_argument("--exp-name", type=str, required=True, help="Specifies a unique experiment name")
parser.add_argument("--trial", type=int, help="The trial number", default=1)
parser.add_argument("--cache-root", type=Path, default=".cachestore", help="Cache directory")
parser.add_argument("--acqf", type=str, help="Acquisition function", choices=["EEIPU", "EIPS", "CArBO", "EI", "RAND"], default="EI")

args, _ = parser.parse_known_args()

disable_cache = args.acqf!="EEIPU"
cache = Cache(f"stacking_{args.exp_name}_{args.trial}_cache", storage=LocalStorage(args.cache_root), disable=disable_cache)

num_samples = 10000 # 9540 seconds to finish 25K samples on 4 gpus

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
@cache(ignore={"output_dir", "dataset", "data_prepoc_output_path"})
def fine_tuning(
    dataset: Union[Path, str],
    data_prepoc_output_path: Union[Path, str],
    output_dir: Union[Path, str],
    *,
    hparams,
):
    """Fine Tuning Stage."""
    dataset = Path(dataset)
    output_dir = Path(output_dir)
    data_prepoc_output_path = Path(data_prepoc_output_path)

    # start_time = time.time()
    model_name = "t5-small"

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["1__learning_rate"], weight_decay=hparams["1__weight_decay"])

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
        hparams["1__num_epochs"],
        tokenizer,
        validation_dataset,
    )

    fine_tuned_model.save_pretrained(output_dir / "fine_tuned_model/")
    fine_tuned_tokenizer.save_pretrained(output_dir / "fine_tuned_tokenizer/")

    # metrics["epoch_num"].append(hparams["1__num_epochs"])
    metrics["learning_rate"].append(hparams["1__learning_rate"])
    metrics["batch_size"].append(hparams["0__batch_size"])

    # end_time = time.time()
    # metrics["cost"] = end_time - start_time

    # with open(output_dir / "metrics.json", "w") as metrics_file:
    #     json.dump(metrics, metrics_file)

    print("fine tuning completed")

    return output_dir


# @task(cache=True, cache_key_file="hparams")
@cache(ignore={"output_dir", "dataset", "fine_tuned_model_path", "data_prepoc_output_path"})
def model_distillation(
    dataset: Union[Path, str],
    data_prepoc_output_path: Union[Path, str],
    fine_tuned_model_path: Union[Path, str],
    output_dir: Union[Path, str],
    *,
    hparams,
):
    """Distillation Stage."""
    output_dir.mkdir(parents=True)
    # start_time = time.time()
    model_name = "t5-small"

    # Load your fine-tuned t5-small teacher model and tokenizer
    teacher_model = T5ForConditionalGeneration.from_pretrained(
        fine_tuned_model_path / "fine_tuned_model/"
    ).to("cuda")

    # Load T5-tiny student model
    tokenizer = T5Tokenizer.from_pretrained(
        model_name, d_model=128, d_ff=512, d_kv=64, num_layers=2
    )
    student_config = T5Config.from_pretrained(
        model_name, d_model=128, d_ff=512, d_kv=64, num_layers=2
    )
    student_model = T5ForConditionalGeneration(student_config).to("cuda")

    optimizer = torch.optim.AdamW(
        student_model.parameters(), lr=hparams["2__learning_rate"], weight_decay=hparams["2__weight_decay"]
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
        hparams["2__num_epochs"],
        tokenizer,
        validation_dataset,
        hparams["2__temperature"],
        output_dir,
    )

    distilled_model.save_pretrained(output_dir / "distilled_model")
    distilled_tokenizer.save_pretrained(output_dir / "distilled_tokenizer")

    # metrics["epoch_num"].append(hparams["2__num_epochs"])
    metrics["learning_rate"].append(hparams["2__learning_rate"])
    metrics["batch_size"].append(hparams["0__batch_size"])
    # end_time = time.time()
    # metrics["cost"] = end_time - start_time
    # with open(output_dir / "metrics.json", "w") as metrics_file:
    #     json.dump(metrics, metrics_file)

    print("Distillation has been completed")

    return metrics["rouge_scores"][0]["rougeLsum"]


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


def select_first_n_stages(stg_hparams: Dict[str, Any], n: int):
    """Select first n stages of the hyperparameters

    n (int): 1 means select only first stage, 2 - second, etc.
    """
    stg_hparams = deepcopy(stg_hparams)
    keys = list(stg_hparams.keys())
    for key in keys:
        if int(key.split("__")[0]) >= n:
            stg_hparams.pop(key)
    return stg_hparams


def t5_fine_tuning(
    dataset: Union[Path, str],
    output_dir: Union[Path, str],
    stg_hparams: List[Dict],
):
    """Main Pipeline."""
    data_preproc_hp = select_first_n_stages(stg_hparams, 1)
    tuning_hp = select_first_n_stages(stg_hparams, 2)
    distillation_hp = select_first_n_stages(stg_hparams, 3)

    start_data_proc = time.time()
    data_prepoc_output_path = data_preprocessing(dataset, output_dir/"data_preprocessing", hparams=data_preproc_hp)
    start_fine_tune = time.time()
    fine_tuned_model_path = fine_tuning(
        dataset, data_prepoc_output_path, output_dir/"fine_tuning", hparams=tuning_hp
    )
    start_distil = time.time()
    rougeLsum = model_distillation(
        dataset,
        data_prepoc_output_path,
        fine_tuned_model_path,
        output_dir/"model_distillation",
        hparams=distillation_hp,
    )
    # inference_output = model_inference(
    #     data_prepoc_output_path,
    #     fine_tuned_model_path,
    #     distilled_model_path,
    #     output_dir,
    #     hparams=stg_hparams,
    # )

    end = time.time()
    # obj = generate_output(
    #     data_prepoc_output_path, fine_tuned_model_path, distilled_model_path
    # )
    stage1_cost = start_fine_tune - start_data_proc
    stage2_cost = start_distil - start_fine_tune
    stage3_cost = end - start_distil

    return {"obj": rougeLsum, "costs": [stage1_cost, stage2_cost, stage3_cost]}


if __name__ == "__main__":
    start = time.time()
    dataset = Path("inputs")
    output_dir = Path("outputs") / time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    stg_hparams = load_hyperparameters(dataset / "hparams.json")
    output = t5_fine_tuning(dataset, output_dir, stg_hparams)
    print(output)
    print("Total duration:", time.time() - start)
