"""blossomtune-llm: A Flower / FlowerTune app."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

FDS = {}  # Cache FederatedDataset


def get_tokenizer(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    return tokenizer


def reformat(dataset):
    """Reformat datasets."""
    dataset = dataset.remove_columns(["dataset_name", "dataset_domain"])
    dataset = dataset.rename_column("question", "prompt")
    dataset = dataset.rename_column("answer", "completion")
    return dataset


def load_data(partition_id: int, num_partitions: int, dataset_name: str, split: str = "train"):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS.get(split) is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS[split] = FederatedDataset(
            dataset=dataset_name,
            partitioners={split: partitioner},
        )
    client_trainset = FDS[split].load_partition(partition_id, split)
    client_trainset = reformat(client_trainset)
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
