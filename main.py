# Imports

from datasets import load_dataset
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import gc
import time
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
import wandb
from trl import SFTTrainer

# Dataset Loading

dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:30%]', 'val[:15%]', 'test[:10%]'])

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


print("Number of samples in training:", len(train_dataset))
print("Number of samples in evaluation:", len(eval_dataset))
print("Number of samples in testing:", len(test_dataset))

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
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]

# Format the datasets

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]

print("Formatted training data:", train_dataset[0])

# Model specifications

model_id = "Qwen/Qwen2-VL-7B-Instruct"

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)

print("Model and processor loaded successfully.")

# Processing the data

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    text_input = processor.apply_chat_template(
        sample[1:2], 
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(sample)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device) 

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]

print("Sample output:", generate_text_from_sample(model, processor, train_dataset[0], device="cuda"))

def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

clear_memory()

# Quantize the model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = Qwen2VLProcessor.from_pretrained(model_id)

# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()

# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2-7b-instruct-trl-sft-ChartQA", 
    num_train_epochs=3, 
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4, 
    gradient_accumulation_steps=8, 
    gradient_checkpointing=True, 
    optim="adamw_torch_fused", 
    learning_rate=1e-4, 
    lr_scheduler_type="constant",  
    logging_steps=10, 
    eval_steps=10, 
    eval_strategy="steps", 
    save_strategy="steps",  
    save_steps=20,  
    metric_for_best_model="eval_loss", 
    greater_is_better=False, 
    load_best_model_at_end=True,  
    bf16=True, 
    tf32=True,  
    max_grad_norm=0.6, 
    warmup_ratio=0.03, 
    push_to_hub=True, 
    report_to="wandb",
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    dataset_text_field="", 
    dataset_kwargs={"skip_prepare_dataset": True}
)

training_args.remove_unused_columns = False

# Initialize Weights and Biases

wandb.init(
    project="Fine Tuning VLM 1", 
    name="Fine Tuning VLM 1", 
    config=training_args
)


def collate_fn(examples):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  
    image_inputs = [process_vision_info(example)[0] for example in examples]  

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  
    labels = batch["input_ids"].clone()  
    labels[labels == processor.tokenizer.pad_token_id] = -100  

    if isinstance(processor, Qwen2VLProcessor):  
        image_tokens = [151652, 151653, 151655] 
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  

    batch["labels"] = labels  

    return batch  

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
)

print("Trainer initialized successfully.")

print("Starting training...")

trainer.train()

print("Training completed.")

print("Saving the model...")

trainer.save_model(training_args.output_dir)

print("Model saved successfully.")

clear_memory()

