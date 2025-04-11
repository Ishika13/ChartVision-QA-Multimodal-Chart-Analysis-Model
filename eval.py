import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

# Load test dataset (using a small subset for quick testing)
dataset_id = "HuggingFaceM4/ChartQA"
test_dataset = load_dataset(dataset_id, split='test[:50]')
print(f"Loaded {len(test_dataset)} test samples")

# System message
system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
Focus on delivering accurate, succinct answers based on the visual information."""

# Format the data
def format_data(sample):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
    ]

# Function to generate answer from model
def generate_answer(model, processor, sample, device="cuda"):
    text_input = processor.apply_chat_template(
        sample, 
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(sample)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device) 

    generated_ids = model.generate(**model_inputs, max_new_tokens=256)

    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0].strip()

# Load base model and processor
print("Loading base model...")
base_model_id = "Qwen/Qwen2-VL-7B-Instruct"
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
base_processor = Qwen2VLProcessor.from_pretrained(base_model_id)

# Load fine-tuned model - but use base processor for both
print("Loading fine-tuned model...")
ft_model_path = "qwen2-7b-instruct-trl-sft-ChartQA"
ft_model = Qwen2VLForConditionalGeneration.from_pretrained(
    ft_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Use the same processor for fine-tuned model
ft_processor = base_processor

# Evaluate models
results = []

print("\nEvaluating models on test set...")
for i, sample in enumerate(test_dataset):
    print(f"\nSample {i+1}/{len(test_dataset)}")
    print(f"Question: {sample['query']}")
    
    # Ground truth
    ground_truth = sample["label"][0].strip()
    print(f"Ground truth: {ground_truth}")
    
    # Format for model input
    formatted_sample = format_data(sample)
    
    # Base model prediction
    base_prediction = generate_answer(base_model, base_processor, formatted_sample)
    print(f"Base model answer: {base_prediction}")
    
    # Fine-tuned model prediction
    ft_prediction = generate_answer(ft_model, ft_processor, formatted_sample)
    print(f"Fine-tuned model answer: {ft_prediction}")
    
    # Check if either model was correct
    base_correct = base_prediction.lower() == ground_truth.lower()
    ft_correct = ft_prediction.lower() == ground_truth.lower()
    
    print(f"Base model correct: {base_correct}")
    print(f"Fine-tuned model correct: {ft_correct}")
    
    results.append({
        "query": sample["query"],
        "ground_truth": ground_truth,
        "base_prediction": base_prediction,
        "ft_prediction": ft_prediction,
        "base_correct": base_correct,
        "ft_correct": ft_correct
    })