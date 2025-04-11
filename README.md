# ChartQA-VLM: Fine-tuned Vision Model for Chart Understanding

## Project Overview
This project fine-tunes the Qwen2-VL-7B-Instruct model to accurately interpret and answer questions about charts and graphs. The fine-tuned model can analyze chart images and provide concise, accurate responses about the data they represent.

## Dataset
The model is trained on the [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) dataset, which contains various chart types (bar, line, pie charts) paired with specific questions and ground truth answers.

## Model Architecture
- **Base Model**: Qwen2-VL-7B-Instruct
- **Optimization**: 
  - 4-bit quantization for memory efficiency
  - LoRA adapters (r=16) targeting attention layers
  - Parameter-efficient fine-tuning (PEFT)

## Training Details
The model was trained for 3 epochs using supervised fine-tuning with the TRL library. Training optimizations include gradient checkpointing, mixed precision (bfloat16), and a learning rate of 1e-4.

## Usage
The repository includes:
- Training script with memory optimization
- Evaluation script comparing fine-tuned vs. base model performance
- Custom data processing for chart images

## Requirements
- GPU with at least 24GB VRAM
- PyTorch, Transformers, PEFT, TRL, and Weights & Biases
