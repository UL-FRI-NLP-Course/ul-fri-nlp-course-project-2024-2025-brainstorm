import os
import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    get_scheduler
)
from transformers import GenerationConfig

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datetime import datetime
import logging
from tqdm import tqdm

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FineTune.data_loader import load_and_prepare_data, create_dataloaders
from FineTune.params_GaMS_9B import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOGGING_DIR}/finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_tokenizer_and_model():
    """Load the tokenizer and model"""
    logger.info(f"Loading tokenizer from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Ensure the tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading model from {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Set up generation config - very important for saving later
    generation_config = GenerationConfig(**GENERATION_CONFIG)
    model.generation_config = generation_config
    
    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        bias=LORA_CONFIG["bias"],
        task_type=LORA_CONFIG["task_type"],
        target_modules=LORA_CONFIG["target_modules"]
    )
    
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Count parameters manually
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return tokenizer, model

def train(model, tokenizer, train_dataloader, val_dataloader):
    """Train the model with custom training loop"""
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    lr_scheduler = get_scheduler(
        name=LR_SCHEDULER,
        optimizer=optimizer,
        num_warmup_steps=int(WARMUP_RATIO * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    # Training loop
    device = model.device
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch+1}/{NUM_EPOCHS}")
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        logger.info(f"Epoch {epoch+1} - Validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-epoch-{epoch+1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        model.train()
    
    # Save final model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"Saved final model to {OUTPUT_DIR}")

def main():
    """Main training function"""
    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)
    
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model()
    
    # Load and prepare data
    csv_path = DARASET_DIR
    train_data, val_data = load_and_prepare_data(csv_path, tokenizer)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_data, val_data, tokenizer, BATCH_SIZE, MAX_SEQ_LENGTH
    )
    
    # Train the model
    train(model, tokenizer, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()