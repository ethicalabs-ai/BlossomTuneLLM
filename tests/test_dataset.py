import pytest
from datasets import Dataset, DatasetDict
from jinja2.exceptions import UndefinedError

from blossomtune.dataset import (
    extract_field_names_from_template,
    reformat_to_messages,
    render_template,
    process_dataset,
)


def test_render_template_valid():
    """Test rendering a valid template."""
    result = render_template("Hello {{ name }}!", {"name": "Alice"})
    assert result == "Hello Alice!"


def test_render_template_missing_context():
    """Test template with missing context variables."""
    with pytest.raises(UndefinedError):
        render_template("Hello {{ name }}!", {})


@pytest.mark.parametrize(
    "template, expected",
    [
        ("Hello {{ name }}!", ["name"]),
        ("{{a}} {{b}} {{c}}", ["a", "b", "c"]),
        ("{{ a }} {{a}} {{ b }}", ["a", "b"]),
        ("No variables", []),
        ("", []),
        ("{{ valid }} {{ valid_with_space }}", ["valid", "valid_with_space"]),
    ],
)
def test_extract_field_names_from_template(template, expected):
    """Test field name extraction from templates."""
    result = extract_field_names_from_template(template)
    assert sorted(result) == sorted(expected)  # Compare sorted lists


def test_reformat_to_messages_success():
    """Test successful reformatting of an example into messages format."""
    example = {"user": "Bob", "response": "Hi!"}
    prompt_tpl = "User: {{ user }}"
    completion_tpl = "Response: {{ response }}"

    updated = reformat_to_messages(example, prompt_tpl, completion_tpl)
    assert updated["messages"] == [
        {"role": "user", "content": "User: Bob"},
        {"role": "assistant", "content": "Response: Hi!"},
    ]


def test_reformat_to_messages_invalid_template(capsys):
    """Test invalid templates print warnings and produce empty messages."""
    example = {"user": "Bob"}
    invalid_tpl = "{{ user | nonexistent_filter }}"

    updated = reformat_to_messages(example, invalid_tpl, invalid_tpl)
    captured = capsys.readouterr()

    assert "Warning: Prompt formatting error" in captured.out
    assert updated["messages"] == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]


def test_process_dataset_with_templates():
    """Test processing a single Dataset with templates produces messages."""
    dataset = Dataset.from_dict({"user": ["Alice", "Bob"]})
    prompt_tpl = "Hello {{ user }}!"
    completion_tpl = "Goodbye {{ user }}!"

    processed = process_dataset(dataset, prompt_tpl, completion_tpl)
    assert processed["messages"] == [
        [
            {"role": "user", "content": "Hello Alice!"},
            {"role": "assistant", "content": "Goodbye Alice!"},
        ],
        [
            {"role": "user", "content": "Hello Bob!"},
            {"role": "assistant", "content": "Goodbye Bob!"},
        ],
    ]


def test_process_dataset_dict_with_templates():
    """Test processing a DatasetDict with templates."""
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"user": ["Alice"]}),
            "test": Dataset.from_dict({"user": ["Bob"]}),
        }
    )
    prompt_tpl = "Hi {{ user }}"
    completion_tpl = "Bye {{ user }}"

    processed = process_dataset(dataset, prompt_tpl, completion_tpl)
    assert processed["train"]["messages"] == [
        [
            {"role": "user", "content": "Hi Alice"},
            {"role": "assistant", "content": "Bye Alice"},
        ],
    ]
    assert processed["test"]["messages"] == [
        [
            {"role": "user", "content": "Hi Bob"},
            {"role": "assistant", "content": "Bye Bob"},
        ],
    ]


def test_process_dataset_conversational_passthrough():
    """Test that a dataset with messages column passes through without templates."""
    messages_data = [
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
    ]
    dataset = Dataset.from_dict({"messages": messages_data})

    processed = process_dataset(dataset)
    assert processed["messages"] == messages_data


def test_process_dataset_no_templates_no_messages():
    """Test that missing templates and no messages column raises an error."""
    dataset = Dataset.from_dict({"text": ["some text"]})
    with pytest.raises(ValueError, match="messages"):
        process_dataset(dataset)


def test_process_dataset_invalid_type():
    """Test passing invalid type raises error."""
    with pytest.raises(TypeError):
        process_dataset("not_a_dataset", "", "")
