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
import json
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
        torch_dtype=MODEL_PRECISION,
        device_map="auto",
        attn_implementation='eager'  # Add this line
    )
    model.init_weights() 
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




def post_process_report(text):
    """Apply rule-based corrections to traffic reports"""
    import re
    
    # Common fixes for Slovenian traffic reports
    corrections = {
        # Passive voice corrections
        r'Cesta je zaprta': 'Zaprta je cesta',
        r'Promet je oviran': 'Oviran je promet',
        r'Avtocesta je zaprta': 'Zaprta je avtocesta',
        
        # Standardize terminology
        r'avtocesta A(\d+)': r'avtocesta A\1',
        r'regionalna cesta R(\d+)': r'regionalna cesta R\1',
        r'glavna cesta G(\d+)': r'glavna cesta G\1',
        
        # Fix common spacing issues
        r'\s+': ' ',  # Multiple spaces to single space
        r'\n\s*\n': '\n\n',  # Normalize paragraph breaks
        r'^\s+': '',  # Remove leading whitespace
        r'\s+$': '',  # Remove trailing whitespace
        
        # Ensure proper punctuation
        r'([^.!?])\s*$': r'\1.',  # Add period at end if missing
        
        # Standardize time format
        r'(\d{1,2})\.(\d{2})\s*ure?': r'\1.\2',
        r'(\d{1,2})\s*ure?': r'\1. ure',
        
        # Fix common traffic terms
        r'zaradi nesreče': 'zaradi prometne nesreče',
    }
    
    cleaned_text = text
    for pattern, replacement in corrections.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)
    
    # Additional cleaning
    cleaned_text = cleaned_text.strip()
    
    # Ensure proper sentence structure
    if cleaned_text and not cleaned_text.endswith(('.', '!', '?')):
        cleaned_text += '.'
    
    return cleaned_text

def generate_report(model, tokenizer, events, timestamp, program_number="2."):
    """Generate a traffic report using the same format as training"""
    import pandas as pd
    import re
    
    # Parse timestamp
    date_obj = pd.to_datetime(timestamp)
    formatted_date = date_obj.strftime("%d. %m. %Y")
    formatted_time = date_obj.strftime("%H.%M")
    
    # Clean HTML tags from events (same as training)
    input_text = re.sub(r'<.*?>', '', str(events))
    input_text = re.sub(r'\s+', ' ', input_text).strip()
    
    # Handle empty inputs (same as training)
    if len(input_text.strip()) < 10:
        input_text = "Ni posebnih prometnih podatkov."
    
    # USE EXACT SAME PROMPT AS TRAINING
    user_message = f"""Si strokovnjak za prometna poročila za slovenski radio. Ustvari kratko prometno poročilo iz podanih podatkov.

    PRAVILA:
    - Pri več dogodkih: najprej najpomembnejše (zapore, nesreče)
    - Uporabljaj kratke stavke, primerne za radio
    - Maksimalno 60 sekund branja
    - Uporabljaj uradni prometni jezik

    STRUKTURA (prilagodi glede na podatke):
    - Glavne zapore/motnje
    - Obvozi
    - Omejitve za tovorna vozila  
    - Vremenske razmere (če vplivajo)

    Podatki o prometu: {input_text}"""
    
    # USE EXACT SAME FORMAT AS TRAINING
    inference_prompt = f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
    
    inputs = tokenizer(inference_prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=300,  # Shorter for radio reports
        temperature=0.3,     # Lower for more consistency
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        num_return_sequences=1
    )
    
    # Decode and extract only the generated part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    report_content = generated_text[len(inference_prompt):].strip()
    
    # Apply rule-based post-processing
    corrected_content = post_process_report(report_content)
    
    # Create header
    header = f"Prometne informacije          {formatted_date}       {formatted_time}         {program_number} program\n\nPodatki o prometu.\n\n"
    
    final_report = header + corrected_content
    
    return final_report, report_content

def test_generation_after_epoch(model, tokenizer, epoch, test_data_path="/d/hpc/projects/onj_fri/brainstorm/not_dataset_folder/not_test_data.csv"):
    """Test generation after each epoch and save results"""
    import pandas as pd
    
    print(f"Running test generation after epoch {epoch+1}")
    
    try:
        # Load test data
        df = pd.read_csv(test_data_path)
        logger.info(f"Loaded {len(df)} test samples")
        
        results = []
        
        # Take only first 10 samples for speed during training
        test_samples = df.head(10)
        
        with torch.no_grad():
            for idx, row in test_samples.iterrows():
                timestamp = row['Datum']
                events = row['Input_porocilo']
                
                # Generate report
                try:
                    final_report, pre_check_report = generate_report(model, tokenizer, events, timestamp)
                    
                    results.append({
                        "sample_id": idx,
                        "timestamp": timestamp,
                        "events": events,
                        "generated_report": final_report,
                        "pre_check_report": pre_check_report,
                        "target_report": row['Porocilo'] if 'Porocilo' in row else None,
                        "epoch": epoch + 1
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate report for sample {idx}: {e}")
                    results.append({
                        "sample_id": idx,
                        "timestamp": timestamp,
                        "events": events,
                        "error": str(e),
                        "epoch": epoch + 1
                    })
        
        # Save results
        job_id = os.environ.get("SLURM_JOB_ID", "local")
        output_file = os.path.join(TESTING_DIR, f"epoch_{epoch+1}_generation_results_{job_id}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved generation results to {output_file}")
        
        # Print a sample for quick inspection
        if results and 'generated_report' in results[0]:
            logger.info(f"Sample generation:\nInput: {results[0]['events'][:100]}...\nOutput: {results[0]['pre_check_report'][:200]}...")
        
          # Set back to training mode
        return results
        
    except Exception as e:
        logger.error(f"Error in test generation: {e}")
        model.train()  # Ensure model is back in training mode
        return []



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
    
    max_grad_norm = 1.0  # Clip gradients at 1.0
    
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
            
            if torch.isnan(loss):
                logger.warning("Skipping batch due to NaN loss")
                continue
            
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} - Average training loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        print(f"Running test generation after epoch {epoch+1}...")
        try:
            test_results = test_generation_after_epoch(model, tokenizer, epoch)
        except Exception as e:
            print(f"Error during test generation: {e}")
        
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