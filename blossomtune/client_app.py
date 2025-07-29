"""blossomtune-llm: A Flower / FlowerTune app."""

import os
import warnings
from typing import Dict, Tuple

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from adopt import ADOPT

from trl import SFTTrainer, SFTConfig

from .config import get_run_config
from .dataset import (
    get_tokenizer,
    load_data,
)
from .models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Standard Flower client for LLM training."""

    def __init__(
        self,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        trainset,
        # valset,
        tokenizer,
        num_rounds,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = SFTConfig(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.num_rounds = num_rounds
        self.trainset = trainset
        # self.valset = valset
        self.model_cfg = model_cfg

        # instantiate model
        self.model = get_model(model_cfg)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        raise NotImplementedError()


# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
class SFTClient(FlowerClient):
    """Standard client for SFT training."""

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        set_parameters(self.model, parameters)

        new_lr = cosine_annealing(
            int(config["current_round"]),
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = config["save_path"]

        if self.model_cfg.use_adopt:
            self.training_arguments.optimizers = (
                ADOPT(self.model.parameters(), lr=new_lr, decouple=True),
                None,
            )
        else:
            self.training_arguments.optimizers = (None, None)

        # Construct trainer
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self.training_arguments,
            train_dataset=self.trainset,
            # eval_dataset=self.valset,
        )

        # Do local training
        results = trainer.train()

        return (
            get_parameters(self.model),
            len(self.trainset),
            {"train_loss": results.training_loss},
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set."""
        raise NotImplementedError()


def client_fn(context: Context) -> FlowerClient:
    """Create a Flower client representing a single organization."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = get_run_config(context)

    # Let's get the client partition
    client_trainset = load_data(
        partition_id,
        num_partitions,
        cfg.dataset.name,
        cfg.dataset.prompt_template,
        cfg.dataset.completion_template,
        "train",
    )
    # client_valset = load_data(
    #     partition_id,
    #     num_partitions,
    #     cfg.dataset.name,
    #     cfg.dataset.prompt_template,
    #     cfg.dataset.completion_template,
    #     "validation",
    # )
    tokenizer = get_tokenizer(cfg.model.name)

    return SFTClient(
        cfg.model,
        cfg.train,
        client_trainset,
        # client_valset,
        tokenizer,
        num_rounds,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
