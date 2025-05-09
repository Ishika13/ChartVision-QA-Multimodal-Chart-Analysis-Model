# ChartVision-QA: Multimodal Chart Analysis Model

## Project Goal
Fine-tune a state-of-the-art Vision-Language Model (VLM) to accurately interpret data visualizations (charts) and answer natural language questions about them, using parameter-efficient techniques and memory optimization strategies.

## Description
ChartVision-QA builds a memory-optimized fine-tuning pipeline for the powerful Qwen2-VL-7B-Instruct model, enabling it to understand and reason over chart images (bar graphs, pie charts, line plots, etc.). Using Low-Rank Adaptation (LoRA) and 4-bit quantization via PEFT and BitsAndBytes, the model is trained on the ChartQA dataset. The result is a scalable, cost-efficient multimodal model that significantly improves visual question answering over structured data representations. A full training and evaluation pipeline is provided, comparing base vs fine-tuned model performance.

## Requirements
- Python >= 3.9
- torch >= 2.0.1
- transformers >= 4.36.2
- datasets >= 2.16.1
- peft >= 0.10.0
- bitsandbytes >= 0.41.2
- wandb

Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

> Note: Make sure you have access to a GPU with at least 16GB memory for efficient training.

## Usage

### Fine-tune the Model
Run the following command to start fine-tuning:
```bash
python main.py
```
This will load the Qwen2-VL-7B-Instruct model, apply LoRA adapters, preprocess the ChartQA dataset, and begin training. Training progress will be logged to Weights & Biases.

### Evaluate the Model
After training, run evaluation to compare the fine-tuned and base model performance:
```bash
python eval.py
```
This will load the fine-tuned model from the `./results` directory and output answers generated by both models side-by-side for a sample of questions from the test dataset.

---

