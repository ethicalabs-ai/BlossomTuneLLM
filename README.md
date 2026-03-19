# BlossomTuneLLM: Federated Supervised Fine-Tune SLMs

> [!NOTE]
> **New Version for Flower Hub!**
> This update brings native support for conversational datasets and performance-optimized disk-based caching, making it easier than ever to fine-tune SLMs in a federated environment.

## Empowering Decentralized & Efficient LLM Customization

BlossomTuneLLM is an open-source project designed to enable **Supervised Fine-Tuning (SFT) of Small Language Models (SLMs)** in a federated learning setup, specifically optimized for **deployment environments leveraging NVIDIA and AMD GPUs**.

Built upon the insights from [flower.ai](https://flower.ai)'s FlowerTune LLM, BlossomTuneLLM significantly enhances the capability to run federated fine-tuning experiments by providing a robust, containerized solution for parallelized workloads.

### Why BlossomTuneLLM:

In an era where large language models demand immense computational resources and often raise privacy concerns due to centralized data processing, this project offers a powerful alternative for:

  * **Decentralization & Privacy-First AI:** Train models collaboratively without centralizing sensitive data.
  * **Efficiency & Sustainability:** Optimize the use of GPU resources by enabling parallelized training and shared disk-based caching.
  * **Accessibility for Small Labs & Researchers:** Provides an accessible framework for smaller research labs, students, and companies to build specialized, privacy-first models.
  * **Customization & Flexibility:** Offers streamlined customization of fine-tuning parameters, target layers, and supports both legacy templates and modern conversational formats.

## Key Features:

  * **Federated Supervised Fine-Tuning (SFT):** Leverages the Flower framework to facilitate federated learning for SLMs.
  * **Conversational Dataset Support:** Native support for TRL's `messages` format, with automated template rendering for backward compatibility.
  * **Performance Optimization:** Disk-based caching of tokenized datasets to eliminate redundant processing across federated rounds.
  * **Deployment-Optimized Execution:** Engineered for real-world deployment leveraging containers and GPU acceleration (NVIDIA/AMD).
  * **Enhanced Customization:** Easily configure fine-tuning parameters, learning rates (with cosine annealing schedule), and specific LoRA target modules.
  * **ADOPT Optimizer Integration:** Optional use of the [ADOPT](https://arxiv.org/abs/2411.02853) optimizer for improved training efficiency.
  * **Apache-2.0 Licensed:** Open and permissive for broad use and collaboration.

## Getting Started with Docker (and Podman):

BlossomTuneLLM is containerized for easy setup and parallelized execution. You can use either Docker or Podman.

---

**Pre-requisite for NVIDIA Container Toolkit:**

If you plan to leverage NVIDIA GPUs with BlossomTuneLLM, you will need to install the NVIDIA Container Toolkit.

This toolkit enables Docker (and Podman via `podman-nvidia-container-runtime`) to interact with your NVIDIA GPUs.

For installation instructions and further information, please refer to the official NVIDIA Container Toolkit documentation: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

First, clone the repository:

```bash
git clone https://github.com/ethicalabs-ai/BlossomTuneLLM.git
cd BlossomTuneLLM
```

### Running Superlink and Supernodes on the Same Machine:

The `docker-compose.yaml` file defines a server node (`blossomtune-server-node`) and multiple client nodes (`blossomtune-client-node-01` to `-04`), each configured to utilize GPU resources.

To bring up all services:

```bash
docker compose up
```

*(Note: If using Podman, you might need to use `podman compose up` or adjust commands based on your Podman setup.)*

### Running the Federated Training:

Once the containers are running, you can execute the federated fine-tuning training.

This example fine-tunes for `SmolLM2-135M-Instruct`.

Access the server node's bash shell:

```bash
docker compose exec -it blossomtune-server-node bash
```

Then, run the Flower training command with your desired `run-config`:

```bash
uv run flwr run --stream --run-config="model.name='HuggingFaceTB/SmolLM2-135M-Instruct' train.training-arguments.per-device-train-batch-size=4 train.training-arguments.bf16=true num-server-rounds=10"
```

This command initiates a federated training run, specifying the model, batch size, mixed-precision training (bf16/tf32), and the number of server rounds.

The fine-tuned model adapter will be available at `./results/huggingfacetb-smollm2-135m-instruct/<DATE_PLACEHOLDER>/peft_100/`.

This allows you to fine-tune multiple models using the same codebase, with unified hyperparameter settings, speeding up research and development.

Additional tooling to merge the adapter and push the merged model to HuggingFace will be provided soon.

## Contributing

We welcome contributions from the community\!

Feel free to open issues, submit pull requests, or join discussions.

## License

BlossomTuneLLM is released under the Apache-2.0 License.
