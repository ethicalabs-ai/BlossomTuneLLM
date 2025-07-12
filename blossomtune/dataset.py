"""blossomtune-llm: A Flower / FlowerTune app."""

import re
from datasets import Dataset, DatasetDict

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from jinja2 import Template, StrictUndefined
from jinja2.exceptions import TemplateError, UndefinedError


FDS = {}  # Cache FederatedDataset


def get_tokenizer(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    return tokenizer


def render_template(template: str, context: dict):
    """Render jinja2 template string."""
    return Template(template, undefined=StrictUndefined).render(**context)


def extract_field_names_from_template(template_string):
    """
    Extracts all unique field names (placeholders) from a template string.
    E.g., "{a} {b} {c}" -> ['a', 'b', 'c']
    """
    if not template_string:  # Handle empty template string
        return []
    pattern = r"\{\{\s*(\w+)\s*\}\}"
    return list(set(re.findall(pattern, template_string)))


def reformat_dynamic(example, prompt_template, completion_template):
    """
    Reformat a single example based on dynamic field names specified in templates.

    Args:
        example (dict): A single example (row) from the dataset.
        prompt_template (str): The template string for the prompt.
        completion_template (str): The template string for the completion.

    Returns:
        dict: The updated example with 'prompt' and 'completion' fields.
    """
    prompt_field_names = extract_field_names_from_template(prompt_template)
    prompt_kwargs = {}
    for field_name in prompt_field_names:
        prompt_kwargs[field_name] = example.get(field_name, "")

    try:
        prompt_value = render_template(prompt_template, prompt_kwargs).strip()
    except (TemplateError, UndefinedError) as e:
        print(
            f"Warning: Prompt formatting error ({e}). Template: '{prompt_template}', Kwargs: {prompt_kwargs}"
        )
        prompt_value = ""

    completion_field_names = extract_field_names_from_template(completion_template)
    completion_kwargs = {}
    for field_name in completion_field_names:
        completion_kwargs[field_name] = example.get(field_name, "")

    try:
        completion_value = render_template(
            completion_template, completion_kwargs
        ).strip()
    except (TemplateError, UndefinedError) as e:
        print(
            f"Warning: Completion formatting error ({e}). Template: '{completion_template}', Kwargs: {completion_kwargs}"
        )
        completion_value = ""

    example["prompt"] = prompt_value
    example["completion"] = completion_value
    return example


def process_dataset_dynamic(dataset, prompt_template, completion_template):
    """
    Apply the dynamic reformat function to a Hugging Face Dataset or DatasetDict.
    """
    if isinstance(dataset, DatasetDict):
        for split in dataset:
            # Use a lambda to pass the templates to the map function
            dataset[split] = dataset[split].map(
                lambda ex: reformat_dynamic(ex, prompt_template, completion_template)
            )
    elif isinstance(dataset, Dataset):
        dataset = dataset.map(
            lambda ex: reformat_dynamic(ex, prompt_template, completion_template)
        )
    else:
        raise TypeError("Input must be a Hugging Face Dataset or DatasetDict.")
    return dataset


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    prompt_template: str,
    completion_template: str,
    split: str = "train",
):
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
    client_trainset = process_dataset_dynamic(
        client_trainset, prompt_template, completion_template
    )
    return client_trainset
