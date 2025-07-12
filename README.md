# BlossomTuneLLM: Federated Supervised Fine-Tuning for Small Language Models (SLMs)

## Empowering Decentralized & Efficient LLM Customization

BlossomTuneLLM is an open-source project designed to enable **Supervised Fine-Tuning (SFT) of Small Language Models (SLMs)** in a federated learning setup, specifically optimized for **deployment environments leveraging NVIDIA GPUs**.

Built upon the insights from [flower.ai](https://flower.ai)'s FlowerTune LLM, BlossomTuneLLM significantly enhances the capability to run federated fine-tuning experiments by providing a robust, containerized solution for parallelized workloads.

### Why Federated Learning Matters:

In an era where large language models demand immense computational resources and often raise privacy concerns due to centralized data processing, this project offers a powerful alternative for:

  * **Decentralization & Privacy-First AI:** Train models collaboratively without centralizing sensitive data, ideal for privacy-critical domains.
  * **Efficiency & Sustainability:** Optimize the use of GPU resources by enabling parallelized training, eliminating waste and fostering more sustainable AI development.
  * **Accessibility for Small Labs & Researchers:** Provides an accessible framework for smaller research labs, students, and companies to build specialized, privacy-first models, reducing dependency on large, proprietary AI services.
  * **Customization & Flexibility:** Offers streamlined customization of fine-tuning parameters, target layers, and includes centralized dataset partitioning from Hugging Face, along with flexible pre-processing templating for prompts and completions.

## Key Features:

  * **Federated Supervised Fine-Tuning (SFT):** Leverages the Flower framework to facilitate federated learning for SLMs.
  * **Deployment-Optimized Execution:** Engineered for real-world deployment by leveraging Docker containers and NVIDIA Container Runtime. This enables parallelized training across multiple nodes, ensuring 100% utilization of GPU resources.
  * **Enhanced Customization:** Easily configure fine-tuning parameters, learning rates (with cosine annealing schedule), and specific LoRA target modules.
  * **Flexible Data Handling:** Supports centralized dataset partitioning from Hugging Face and dynamic prompt/completion templating for diverse datasets.
  * **ADOPT Optimizer Integration:** Optional use of ADOPT optimizer for improved training efficiency.
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
uv run flwr run . local-deployment --stream --run-config="use-flexlora=false strategy.fraction-fit=1 model.name='HuggingFaceTB/SmolLM2-135M-Instruct' train.training-arguments.per-device-train-batch-size=8 train.training-arguments.bf16=true train.training-arguments.tf32=true num-server-rounds=100"
```

This command initiates a federated training run, specifying the model, batch size, mixed-precision training (bf16/tf32), and the number of server rounds.

## Contributing

We welcome contributions from the community\!

Feel free to open issues, submit pull requests, or join discussions.

## License

BlossomTuneLLM is released under the Apache-2.0 License.
