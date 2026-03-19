# BlossomTuneLLM: Federated Fine-Tuning Simulation Setup for SLMs using Flower and TRL

BlossomTuneLLM is a streamlined setup for simulating **Federated Supervised Fine-Tuning (SFT)** of Small Language Models (SLMs).

Optimized for **Flower 1.27.0+** and **[Flower Apps](https://flower.ai/apps/)** readiness, it prioritizes **AMD ROCm** performance while remaining compatible with NVIDIA CUDA.

## Getting Started (AMD ROCm / Ubuntu)

We recommend using [uv](https://github.com/astral-sh/uv) to manage your environment directly.

### 1. Installation
```bash
git clone https://github.com/ethicalabs-ai/BlossomTuneLLM.git
cd BlossomTuneLLM

# Install dependencies with ROCm support
uv sync --group rocm
```

### 2. Host Configuration
Flower uses a local configuration file to define simulation resources. Create or update `~/.flwr/config.toml` on your host machine:

```toml
[superlink]
default = "local"

[superlink.supergrid]
address = "supergrid.flower.ai"

[superlink.local]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 4
options.backend.client-resources.num-gpus = 1
```

### 3. Run Simulation
Initiate the federated training simulation across all partitions:
```bash
uv run flwr run --stream
```

To override the model or change the number of rounds:
```bash
uv run flwr run --stream --run-config="model.name='HuggingFaceTB/SmolLM2-135M-Instruct' num-server-rounds=5"
```

## Running with Docker (Optional)

If you prefer containerized execution:

1. **Startup**: `docker compose up -f docker-compose-rocm.yaml`
2. **Execute**: `docker compose exec -it blossomtune-server-node bash`
3. **Run**: `uv run flwr run --stream`

## Key Features
- **AMD ROCm Priority**: Native optimization for ROCm 6.x environments.
- **Flower Hub Ready**: Architected for seamless integration into **[Flower Apps](https://flower.ai/apps/)**.
- **Conversational Support**: Native handling of TRL's `messages` format with automated template fallback.
- **Efficient Caching**: Disk-based tokenization caching in `./data/cache` to eliminate redundant processing.

## Configuration
Customize your simulation in `pyproject.toml` under `[tool.flwr.app.config]`:

### Dataset & Hardware
- `dataset.name`: Target Hugging Face dataset (e.g., `mlabonne/FineTome-100k-dedup`).
- `data-path`: Directory for shared caching (default: `./data`).
- `save-path`: Where results and adapters are saved.

### Model & Quantization
- `model.name`: Target Hugging Face model (e.g., `HuggingFaceTB/SmolLM2-135M-Instruct`).
- `model.quantization`: Bits for quantization (default: `4`).
- `model.use-adopt`: Whether to use the ADOPT optimizer.

### LoRA (Low-Rank Adaptation)
- `model.lora.peft-lora-r`: LoRA rank.
- `model.lora.peft-lora-alpha`: LoRA alpha.
- `model.lora.peft-target-modules`: Modules to apply LoRA (e.g., `qkv_proj,out_proj`).

### Training Parameters
- `train.seq-length`: Maximum sequence length (default: `4096`).
- `train.learning-rate-max`: Peak learning rate.
- `num-server-rounds`: Total federated learning rounds.
- `train.training-arguments.*`: Direct mapping to `SFTConfig` parameters (batch size, epochs, etc.).

---
*BlossomTuneLLM is a simulation setup for federated learning, released under the Apache-2.0 License. It is intended for research and educational purposes only.*
