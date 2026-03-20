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
3. **Run**: 

```bash
uv run flwr run --stream
```

---

## Sample Run (SmolLM2-135M-Instruct on FineTome-100k-dedup)

Below are the results from a sample simulation run using the default configuration (`SmolLM2-135M-Instruct` on `FineTome-100k-dedup`).

| Metric | Initial (Round 1) | Final (Round 20) |
| :--- | :--- | :--- |
| **Mean Token Accuracy** | 0.6194 | 0.6989 |
| **Training Loss** | 1.5634 | 1.3144 |

- **Total Rounds**: 20
- **Total Time**: 509.59s (~8.5 minutes)
- **Communication Cost**: 798.93 MB

### Logs

```
INFO :      aggregate_fit: received 2 results and 0 failures
INFO :      Communication budget: used 798.93 MB (+19.97 MB this round) / 200,000 MB
Loading weights: 100%|██████████| 272/272 [00:00<00:00, 2236.70it/s, Materializing param=model.norm.weight]                              
INFO :      fit progress: (20, 0.0, {}, 509.58936939199975)
INFO :      configure_evaluate: no clients selected, skipping evaluation
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 20 round(s) in 509.59s
INFO :          History (loss, centralized):
INFO :                  round 0: 0.0
INFO :                  round 1: 0.0
INFO :                  round 2: 0.0
INFO :                  round 3: 0.0
INFO :                  round 4: 0.0
INFO :                  round 5: 0.0
INFO :                  round 6: 0.0
INFO :                  round 7: 0.0
INFO :                  round 8: 0.0
INFO :                  round 9: 0.0
INFO :                  round 10: 0.0
INFO :                  round 11: 0.0
INFO :                  round 12: 0.0
INFO :                  round 13: 0.0
INFO :                  round 14: 0.0
INFO :                  round 15: 0.0
INFO :                  round 16: 0.0
INFO :                  round 17: 0.0
INFO :                  round 18: 0.0
INFO :                  round 19: 0.0
INFO :                  round 20: 0.0
INFO :          History (metrics, distributed, fit):
INFO :          {'mean_token_accuracy': [(1, 0.6193887591362),
INFO :                                   (2, 0.6945021450519562),
INFO :                                   (3, 0.621947569318286),
INFO :                                   (4, 0.6244904100894928),
INFO :                                   (5, 0.6439206004142761),
INFO :                                   (6, 0.6315028667449951),
INFO :                                   (7, 0.6151145398616791),
INFO :                                   (8, 0.6488686203956604),
INFO :                                   (9, 0.6545383036136627),
INFO :                                   (10, 0.6947891639034163),
INFO :                                   (11, 0.6794840693473816),
INFO :                                   (12, 0.6460303457889447),
INFO :                                   (13, 0.6458145976066589),
INFO :                                   (14, 0.6454960211381574),
INFO :                                   (15, 0.6434220798798346),
INFO :                                   (16, 0.641651408395645),
INFO :                                   (17, 0.6557334959506989),
INFO :                                   (18, 0.6388434865791515),
INFO :                                   (19, 0.6311497000616658),
INFO :                                   (20, 0.6988894457045706)],
INFO :           'train_loss': [(1, 1.5634249091148376),
INFO :                          (2, 1.3826697766780853),
INFO :                          (3, 1.2980086541343532),
INFO :                          (4, 1.429375284910202),
INFO :                          (5, 1.2049865543842317),
INFO :                          (6, 1.255205899477005),
INFO :                          (7, 1.273926594853401),
INFO :                          (8, 1.1536943286657335),
INFO :                          (9, 1.1846310704946519),
INFO :                          (10, 1.221038764893365),
INFO :                          (11, 1.1642542511224747),
INFO :                          (12, 1.2229050550515315),
INFO :                          (13, 1.1506917148828508),
INFO :                          (14, 1.1165486176416402),
INFO :                          (15, 1.1006271340861389),
INFO :                          (16, 1.1286018041137833),
INFO :                          (17, 1.1299030184745789),
INFO :                          (18, 1.2965837746982867),
INFO :                          (19, 1.2364309805968514),
INFO :                          (20, 1.3144098730340013)]}
```

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
- `model.lora.peft-target-modules`: Modules to apply LoRA (e.g., `q_proj,v_proj`).

### Training Parameters
- `train.seq-length`: Maximum sequence length (default: `4096`).
- `train.learning-rate-max`: Peak learning rate.
- `num-server-rounds`: Total federated learning rounds.
- `train.training-arguments.*`: Direct mapping to `SFTConfig` parameters (batch size, epochs, etc.).

---
*BlossomTuneLLM is a simulation setup for federated learning, released under the Apache-2.0 License. It is intended for research and educational purposes only.*
