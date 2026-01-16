# Llama 3.1 (8B) GRPO Fine-tuning with Unsloth

This repository contains a Jupyter Notebook for fine-tuning the **Llama 3.1 8B Instruct** model using **Group Relative Policy Optimization (GRPO)**. The training pipeline is optimized using [Unsloth](https://github.com/unslothai/unsloth) to run efficiently on free GPU tiers (like Google Colab's Tesla T4).

## 🚀 Overview

The goal of this project is to enhance the reasoning capabilities of Llama 3.1 by using Reinforcement Learning (RL) techniques—specifically GRPO. Unlike standard Supervised Fine-Tuning (SFT), GRPO generates multiple outputs for a given prompt and uses a set of reward functions to optimize the model's policy towards desired behaviors (correctness, format adherence, etc.).

## ✨ Key Features

* **Model:** Llama 3.1 8B Instruct (4-bit quantized for memory efficiency).
* **Method:** Group Relative Policy Optimization (GRPO).
* **Optimization:** Uses **Unsloth** for 2x faster training and ~60% less memory usage.
* **Dataset:** GSM8K (Grade School Math) used for training chain-of-thought reasoning.
* **Reward Functions:** Custom reward logic implementing:
    * **Correctness:** Verifies mathematical accuracy.
    * **Format:** Enforces strict XML structuring (`<reasoning>` and `<answer>` tags).
    * **Integer Check:** Ensures numerical outputs.

## 🛠️ Prerequisites

To run this notebook, you need:

* **Python 3.10+**
* **GPU:** NVIDIA Tesla T4 or better (fully compatible with Google Colab Free Tier).

The notebook handles the installation of key libraries, including:
* `unsloth`
* `vllm`
* `transformers`
* `torch`
* `trl`

## 📖 Usage

### Running on Google Colab
1.  Upload `Llama3.1_(8B)-GRPO.ipynb` to Google Colab.
2.  Ensure your runtime is set to **T4 GPU** (Runtime > Change runtime type).
3.  Run all cells sequentially.

### Workflow Summary
The notebook proceeds through the following stages:
1.  **Environment Setup:** Installs Unsloth and vLLM dependencies.
2.  **Model Loading:** Loads the base Llama 3.1 model in 4-bit mode with LoRA adapters.
3.  **Data Preparation:** Downloads and formats the GSM8K dataset.
4.  **Reward Definition:** Establishes the "grading rubric" for the model's outputs.
5.  **Training:** Executes the `GRPOTrainer` loop.
6.  **Inference:** Tests the model's ability to reason through new problems.
7.  **Saving/Exporting:** Exports the fine-tuned LoRA adapters or merges them to GGUF for local use (e.g., in Ollama).

## 📊 Training Configuration

* **LoRA Rank (r):** 16
* **LoRA Alpha:** 16
* **Optimizer:** AdamW (8-bit)
* **Scheduler:** Cosine
* **Precision:** bfloat16 (on supported hardware) or float16

## 🙏 Acknowledgements

* **[Unsloth AI](https://unsloth.ai/)**: For the optimized training framework.
* **[Hugging Face](https://huggingface.co/)**: For the ecosystem and dataset hosting.
