"""library imports."""
import json
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
from pathlib import Path, PosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union

import boto3
import evaluate
import psutil
import torch
import s3fs
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.parallel import DataParallel
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase

s3 = s3fs.S3FileSystem(config_kwargs = dict(region_name="me-central-1"))

def calculate_bleu_score(predictions, references):
    """Function for calculating bleu score."""
    return corpus_bleu(
        [[ref.split()] for ref in references], [pred.split() for pred in predictions]
    )


def load_hyperparameters(hparams_path: Union[Path, str]) -> Dict:
    """Function for loading hyperparameters.
    hparams_path (Path or str): directory path where the hyperparameters json file is stored
    """
    with s3.open(hparams_path) as _hparams:
        hyperparmeters = json.load(_hparams)
    hyperparameters_dict = hyperparmeters["new_hp"]
    return hyperparameters_dict


def preprocess_dataset(dataset, tokenizer, max_input_length, max_summary_length):
    """Function for data preprocessing."""
    inputs = [f"summarize: {article}" for article in dataset["article"]]
    summaries = dataset["highlights"]
    inputs_encodings = tokenizer(
        inputs,
        truncation=True,
        padding=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    summaries_encodings = tokenizer(
        summaries,
        truncation=True,
        padding=True,
        max_length=max_summary_length,
        return_tensors="pt",
    )
    return inputs_encodings, summaries_encodings


def model_validation(model, val_dataloader, generated_summaries):
    """Function for model validation used in all stages."""
    model.eval()
    with torch.no_grad():
        val_losses = []
        for val_batch in val_dataloader:
            (
                val_input_ids,
                val_input_attention_mask,
                val_summary_ids,
                val_summary_attention_mask,
            ) = val_batch

            val_outputs = model(
                input_ids=val_input_ids,
                attention_mask=val_input_attention_mask,
                labels=val_summary_ids,
            )
            val_loss = val_outputs.loss
            val_loss = val_loss.mean()
            val_losses.append(val_loss.item())

            for logits in val_outputs.logits:
                generated_summary = logits.argmax(dim=-1)
                generated_summaries.append(generated_summary)

    return val_losses, generated_summaries


def tuning(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs,
    tokenizer,
    validation_dataset,
    patience=3,  # Number of consecutive epochs without improvement before stopping
    accumulation_steps=2,
) -> Tuple[Dict[str, List], PreTrainedModel, PreTrainedTokenizerBase]:
    """Function for fine tuning."""
    torch.cuda.empty_cache()
    model = DataParallel(model)
    rouge_metric = evaluate.load("rouge")
    metrics = {
        "training_loss": [],
        "validation_loss": [],
        "rouge_scores": [],
        "bleu_scores": [],
        "epoch_num": [],
        "learning_rate": [],
        "batch_size": [],
        "cost": 0,
    }
    model.train()

    best_val_loss = float("inf")
    consecutive_no_improvement = 0
    # generated_summaries = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        generated_summaries = []
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            (
                input_ids,
                input_attention_mask,
                summary_ids,
                summary_attention_mask,
            ) = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_attention_mask,
                labels=summary_ids,
            )
            loss = outputs.loss
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_losses, generated_summaries = model_validation(
            model, val_dataloader, generated_summaries
        )

        avg_loss = total_loss / len(train_dataloader)
        metrics["training_loss"].append(avg_loss)
        avg_val_loss = sum(val_losses) / len(val_losses)
        metrics["validation_loss"].append(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            print(f"Early stopping - epoch {epoch + 1} with no improvement.")
            break

    generated_summaries = [
        tokenizer.decode(summary, skip_special_tokens=True)
        for summary in generated_summaries
    ]
    rouge_metric.add_batch(
        predictions=generated_summaries,
        references=validation_dataset["highlights"],
    )
    rouge_scores = rouge_metric.compute()
    metrics["rouge_scores"].append(rouge_scores)

    bleu_score = calculate_bleu_score(
        generated_summaries, validation_dataset["highlights"]
    )
    metrics["bleu_scores"].append(bleu_score)

    model = model.module
    return metrics, model, tokenizer


def distillation(
    teacher_model,
    student_model,
    train_dataloader,
    val_dataloader,
    optimizer,
    num_epochs,
    tokenizer,
    validation_dataset,
    temperature,
    output_dir,
    patience=3,  # Number of consecutive epochs without improvement before stopping
    accumulation_steps=2,
) -> Tuple[Dict[str, List], PreTrainedModel, PreTrainedTokenizerBase]:
    """Function for model distillation."""
    log_file = s3.open(output_dir / "logs.txt", "a", encoding="utf-8")
    log_file.write("starting logs\n")
    torch.cuda.empty_cache()
    rouge_metric = evaluate.load("rouge")
    student_model = DataParallel(student_model)
    log_file.write("scalar done")
    metrics = {
        "training_loss": [],
        "validation_loss": [],
        "rouge_scores": [],
        "bleu_scores": [],
        "epoch_num": [],
        "learning_rate": [],
        "batch_size": [],
        "cost": 0,
    }

    best_val_loss = float("inf")
    consecutive_no_improvement = 0

    for epoch in range(num_epochs):
        student_model.train()
        total_loss = 0.0
        generated_summaries = []
        batch_count = 0
        for batch in train_dataloader:
            (input_ids, attention_mask, summary_ids, summary_attention_mask) = batch
            (input_ids, attention_mask, summary_ids, summary_attention_mask) = (
                input_ids.to("cuda"),
                attention_mask.to("cuda"),
                summary_ids.to("cuda"),
                summary_attention_mask.to("cuda"),
            )
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=summary_ids,
                ).logits
            student_logits = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=summary_ids,
            ).logits
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        val_losses, generated_summaries = model_validation(
            student_model, val_dataloader, generated_summaries
        )

        avg_loss = total_loss / len(train_dataloader)
        metrics["training_loss"].append(avg_loss)
        avg_val_loss = sum(val_losses) / len(val_losses)
        metrics["validation_loss"].append(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= patience:
            print(f"Early stopping - epoch {epoch + 1} with no improvement.")
            break
        log_file.write(f"Epoch Completed: {epoch}\n")
    generated_summaries = [
        tokenizer.decode(summary, skip_special_tokens=True)
        for summary in generated_summaries
    ]
    rouge_metric.add_batch(
        predictions=generated_summaries,
        references=validation_dataset["highlights"],
    )
    rouge_scores = rouge_metric.compute()
    metrics["rouge_scores"].append(rouge_scores)

    bleu_score = calculate_bleu_score(
        generated_summaries, validation_dataset["highlights"]
    )
    metrics["bleu_scores"].append(bleu_score)
    student_model = student_model.module
    return metrics, student_model, tokenizer


def quantization(model, val_dataloader, validation_dataset, tokenizer, output_dir):
    """Function for model quantization."""
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    rouge_metric = evaluate.load("rouge")
    generated_summaries = []
    metrics = {"validation_loss": [], "rouge_scores": [], "bleu_scores": []}
    val_losses, generated_summaries = model_validation(
        model, val_dataloader, generated_summaries
    )
    avg_val_loss = sum(val_losses) / len(val_losses)
    metrics["validation_loss"].append(avg_val_loss)

    generated_summaries = [
        tokenizer.decode(summary, skip_special_tokens=True)
        for summary in generated_summaries
    ]
    rouge_metric.add_batch(
        predictions=generated_summaries,
        references=validation_dataset["highlights"],
    )
    rouge_scores = rouge_metric.compute()
    metrics["rouge_scores"].append(rouge_scores)

    # Calculate BLEU score
    bleu_score = calculate_bleu_score(
        generated_summaries, validation_dataset["highlights"]
    )
    metrics["bleu_scores"].append(bleu_score)

    return model, metrics


def inference(model, tokenizer, test_data, reference_summaries):
    metrics = {
        "average_memory_usage": 0,
        "average_inference_time": 0,
    }

    memory_usage_total = 0  # Initialize total memory usage
    inference_time_total = 0  # Initialize total inference time

    for text in test_data:
        # Tokenize the input text
        input_ids = tokenizer.encode(text, return_tensors="pt")

        # Track memory usage before inference
        process = psutil.Process(os.getpid())
        start_memory_usage = process.memory_info().rss / 1024  # in KB

        start_time = time.time()

        with torch.no_grad():
            _ = model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True,
            )

        end_time = time.time()

        # Track memory usage after inference
        end_memory_usage = process.memory_info().rss / 1024  # in KB
        memory_usage_change = end_memory_usage - start_memory_usage

        # Update total memory usage and total inference time
        memory_usage_total += memory_usage_change
        inference_time_total += end_time - start_time

    # Calculate average memory usage
    metrics["average_memory_usage"] = memory_usage_total / len(test_data)

    # Calculate average inference time
    metrics["average_inference_time"] = inference_time_total / len(test_data)

    return metrics


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

  
@contextmanager 
def s3_fileobj(file_path): 
    """
    Yields a file object from the filename at {bucket}/{key}

    Args:
        bucket (str): Name of the S3 bucket where you model is stored
        key (str): Relative path from the base of your bucket, including the filename and extension of the object to be retrieved.
    """
    with s3.open(file_path) as f:
        yield BytesIO(f.read()) 


def download_model(path_to_model, tmp_dir):
    """
    Download model at the given S3 path.
    """
    s3_files = s3.listdir(path_to_model, detail=False)
    print("Files to download:", s3_files)
    for s3_file in s3_files:
        with s3_fileobj(s3_file) as f: 
            with open(f"{tmp_dir}/{Path(s3_file).name}", "wb") as tempfile:
                tempfile.write(f.read()) 

def upload_model(local_path_to_model, s3_path_to_model):
    """Upload saved model to s3
    
    Keyword arguments:
    local_path_to_model -- local path where model is saved. Likely a temporary directory
    s3_path_to_model -- path to s3 storage
    Return: None
    """
    files = os.listdir(local_path_to_model)
    print("Files to upload:", files)
    for file in files:
        with s3.open(f"{s3_path_to_model}/{file}", "wb") as f: 
            with open(f"{local_path_to_model}/{file}", "rb") as tempfile:
                f.write(tempfile.read()) 

def torch_save(obj, path_to_model, **kwargs):
    """Function to save pytorch object to S3"""
    if isinstance(path_to_model, Path):
        path_to_model = path_to_model.as_posix()
    return torch.save(obj, s3.open(path_to_model, "wb"))

class AWSPath(Path):
    _flavour = PosixPath._flavour
    def exists(self) -> bool:
        return s3.exists(self.as_posix())
    def mkdir(self, mode: int = 511, parents: bool = False, exist_ok: bool = False) -> None:
        return
    def open(self, mode='r', buffering=-1, encoding=None,
             errors=None, newline=None):
        return s3.open(self.as_posix(), mode)
    
