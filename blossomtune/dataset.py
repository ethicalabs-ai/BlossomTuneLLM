"""blossomtune-llm: A Flower / FlowerTune app."""

import re
from datasets import Dataset, DatasetDict

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer

FDS = {}  # Cache FederatedDataset


def get_tokenizer(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    return tokenizer


def extract_field_names_from_template(template_string):
    """
    Extracts all unique field names (placeholders) from a template string.
    E.g., "{a} {b} {c}" -> ['a', 'b', 'c']
    """
    if not template_string:  # Handle empty template string
        return []
    return list(set(re.findall(r"\{(\w+)\}", template_string)))


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
        prompt_value = prompt_template.format(**prompt_kwargs).strip()
    except KeyError as e:
        print(
            f"Warning: Prompt formatting error. Missing key: {e}. Template: '{prompt_template}', Kwargs: {prompt_kwargs}"
        )
        prompt_value = ""

    completion_field_names = extract_field_names_from_template(completion_template)
    completion_kwargs = {}
    for field_name in completion_field_names:
        completion_kwargs[field_name] = example.get(field_name, "")

    try:
        completion_value = completion_template.format(**completion_kwargs).strip()
    except KeyError as e:
        print(
            f"Warning: Completion formatting error. Missing key: {e}. Template: '{completion_template}', Kwargs: {completion_kwargs}"
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


if __name__ == "__main__":
    # Define your templates directly
    my_prompt_template = "{a} {b} {c} asdsadasd asda d"
    my_completion_template = "{output} - {score} for task: {task_id}"

    # Create a dummy dataset with corresponding fields
    data_sample = {
        "a": ["First A", "Second A"],
        "b": ["First B", "Second B"],
        "c": ["First C", "Second C"],
        "output": ["Output 1", "Output 2"],
        "score": [95, 88],
        "task_id": ["TaskX", "TaskY"],
    }
    dataset_sample = Dataset.from_dict(data_sample)

    print("--- Processing dataset_sample ---")
    processed_dataset = process_dataset_dynamic(
        dataset_sample, my_prompt_template, my_completion_template
    )
    print("Example 0:")
    print(f"  Original: {dataset_sample[0]}")
    print(f"  Prompt: '{processed_dataset[0]['prompt']}'")
    print(f"  Completion: '{processed_dataset[0]['completion']}'")

    print("\nExample 1:")
    print(f"  Original: {dataset_sample[1]}")
    print(f"  Prompt: '{processed_dataset[1]['prompt']}'")
    print(f"  Completion: '{processed_dataset[1]['completion']}'")

    # --- Example with missing fields in dataset ---
    print("\n--- Example with some fields missing in dataset ---")
    data_missing = {
        "a": ["Only A here"],
        "output": ["Simple Output"],
        # 'b', 'c', 'score', 'task_id' are missing
    }
    dataset_missing = Dataset.from_dict(data_missing)

    processed_missing = process_dataset_dynamic(
        dataset_missing, my_prompt_template, my_completion_template
    )
    print("Example 0 (missing fields):")
    print(f"  Original: {dataset_missing[0]}")
    print(f"  Prompt: '{processed_missing[0]['prompt']}'")  # 'b' and 'c' will be empty
    print(
        f"  Completion: '{processed_missing[0]['completion']}'"
    )  # 'score' and 'task_id' will be empty

    # --- Example with empty templates ---
    print("\n--- Example with empty templates ---")
    data_empty_templates = {"any_field": ["Data"]}
    dataset_empty_templates = Dataset.from_dict(data_empty_templates)

    processed_empty_templates = process_dataset_dynamic(dataset_empty_templates, "", "")
    print("Example 0 (empty templates):")
    print(f"  Original: {dataset_empty_templates[0]}")
    print(f"  Prompt: '{processed_empty_templates[0]['prompt']}'")
    print(f"  Completion: '{processed_empty_templates[0]['completion']}'")
