"""blossomtune-llm: A Flower / FlowerTune app."""

import re
import os
import hashlib
from datasets import Dataset, DatasetDict, load_from_disk
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from jinja2 import Template, StrictUndefined
from jinja2.exceptions import TemplateError, UndefinedError


FDS = {}  # Cache FederatedDataset
TOKENIZED_CACHE = {}  # Cache tokenized datasets across rounds (in-memory)


def get_tokenizer(model_name: str):
    """Get tokenizer, data_collator and prompt formatting."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
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


def reformat_to_messages(example, prompt_template, completion_template):
    """
    Reformat a single example into TRL's messages format using templates.

    Renders prompt and completion templates, then builds a messages list
    with user/assistant roles for TRL's SFTTrainer.

    Args:
        example (dict): A single example (row) from the dataset.
        prompt_template (str): The template string for the user message.
        completion_template (str): The template string for the assistant message.

    Returns:
        dict: The updated example with a 'messages' field.
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

    example["messages"] = [
        {"role": "user", "content": prompt_value},
        {"role": "assistant", "content": completion_value},
    ]
    return example


def map_conversations_to_messages(example):
    """
    Map ShareGPT 'conversations' format to TRL 'messages' format.
    Standardizes Roles:
    - human/user -> user
    - gpt/assistant/bot/chatgpt -> assistant
    - system -> system
    """
    if "conversations" in example:
        messages = []
        for turn in example["conversations"]:
            role = turn.get("from", "")
            content = turn.get("value", "")
            if not role or not content:
                continue
            if role in ["human", "user"]:
                role = "user"
            elif role in ["gpt", "assistant", "bot", "chatgpt"]:
                role = "assistant"
            elif role == "system":
                role = "system"
            else:
                # Fallback: assume human if unknown, or skip?
                # For now, let's skip unknown roles to avoid ambiguity
                continue
            messages.append({"role": role, "content": content})
        example["messages"] = messages
    return example


def process_dataset(dataset, prompt_template="", completion_template=""):
    """
    Process a dataset into TRL's messages format.

    If templates are provided, renders them to build messages.
    If templates are empty, expects the dataset to already have a 'messages' column.
    """
    if not isinstance(dataset, (Dataset, DatasetDict)):
        raise TypeError("Input must be a Hugging Face Dataset or DatasetDict.")

    has_templates = bool(prompt_template) and bool(completion_template)

    def _get_columns(ds):
        if isinstance(ds, DatasetDict):
            first_split = next(iter(ds))
            return ds[first_split].column_names
        return ds.column_names

    if not has_templates:
        columns = _get_columns(dataset)
        if "messages" not in columns:
            if "conversations" in columns:
                # Handle ShareGPT format automatically
                if isinstance(dataset, DatasetDict):
                    for split in dataset:
                        dataset[split] = dataset[split].map(
                            map_conversations_to_messages,
                            desc="Mapping conversations to messages",
                        )
                else:
                    dataset = dataset.map(
                        map_conversations_to_messages,
                        desc="Mapping conversations to messages",
                    )
            else:
                raise ValueError(
                    "No prompt/completion templates provided and dataset does not "
                    "contain a 'messages' or 'conversations' column. Either provide "
                    "templates or use a conversational dataset."
                )
        # Dataset already has messages (or was just mapped) — pass through
        return dataset

    # Legacy path: build messages from templates
    if isinstance(dataset, DatasetDict):
        for split in dataset:
            dataset[split] = dataset[split].map(
                lambda ex: reformat_to_messages(
                    ex, prompt_template, completion_template
                )
            )
    elif isinstance(dataset, Dataset):
        dataset = dataset.map(
            lambda ex: reformat_to_messages(ex, prompt_template, completion_template)
        )
    return dataset


def _tokenize_dataset(dataset, tokenizer, max_seq_length):
    """Tokenize a dataset with messages column using the tokenizer's chat template.

    Returns a tokenized dataset with input_ids, attention_mask, and labels.
    """

    def tokenize_fn(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(
        tokenize_fn,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )
    return tokenized


def _get_cache_path(
    data_path,
    dataset_name,
    partition_id,
    split,
    prompt_template,
    completion_template,
    tokenizer_name,
    max_seq_length,
):
    """Generate a unique disk cache path for the tokenized dataset."""
    abs_data_path = os.path.abspath(data_path)
    cache_dir = os.path.join(abs_data_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    config_str = f"{dataset_name}_{partition_id}_{split}_{prompt_template}_{completion_template}_{tokenizer_name}_{max_seq_length}"
    config_hash = hashlib.md5(config_str.encode()).hexdigest()

    return os.path.join(cache_dir, f"tokenized_{config_hash}")


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str,
    prompt_template: str = "",
    completion_template: str = "",
    split: str = "train",
    tokenizer=None,
    max_seq_length: int = 4096,
    data_path: str = "./data",
):
    """Load partition data.

    If tokenizer is provided, the dataset will be tokenized once and cached
    to avoid repeated tokenization across federated rounds.
    """
    global FDS, TOKENIZED_CACHE

    cache_key = (partition_id, split, dataset_name)

    # Return cached tokenized dataset if available in memory
    if cache_key in TOKENIZED_CACHE:
        return TOKENIZED_CACHE[cache_key]

    # Check for disk cache if tokenizer is provided
    cache_path = None
    if tokenizer is not None:
        tokenizer_name = getattr(tokenizer, "name_or_path", "unknown")
        cache_path = _get_cache_path(
            data_path,
            dataset_name,
            partition_id,
            split,
            prompt_template,
            completion_template,
            tokenizer_name,
            max_seq_length,
        )
        if os.path.exists(cache_path):
            print(f"Loading tokenized dataset from disk cache: {cache_path}")
            tokenized_dataset = load_from_disk(cache_path)
            TOKENIZED_CACHE[cache_key] = tokenized_dataset
            return tokenized_dataset

    # Only initialize `FederatedDataset` once
    if FDS.get(split) is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        FDS[split] = FederatedDataset(
            dataset=dataset_name,
            partitioners={split: partitioner},
        )
    client_trainset = FDS[split].load_partition(partition_id, split)
    client_trainset = process_dataset(
        client_trainset, prompt_template, completion_template
    )

    # Tokenize and cache if tokenizer provided
    if tokenizer is not None:
        client_trainset = _tokenize_dataset(client_trainset, tokenizer, max_seq_length)
        if cache_path:
            print(f"Saving tokenized dataset to disk cache: {cache_path}")
            client_trainset.save_to_disk(cache_path)
        TOKENIZED_CACHE[cache_key] = client_trainset

    return client_trainset
