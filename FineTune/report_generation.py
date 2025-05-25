import re
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from datetime import datetime
import json

# Import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FineTune.params_GaMS_9B import *


#! Set path to the desierd model
PATH_TO_MODEL ="/d/hpc/projects/onj_fri/brainstorm/FineTune_models/GaMS-9B-Instruct/checkpoints/checkpoint-epoch-1"

# TEST DATASET PATH
PATH_TO_DATASET = "/d/hpc/projects/onj_fri/brainstorm/not_dataset_folder/not_test_data.csv"

# PROMPT FOR GENERATING REPORTS
PROMPT = """Si strokovnjak za prometna poročila za slovenski radio. Ustvari kratko prometno poročilo iz podanih podatkov.

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

    Podatki o prometu: """

# REPORT HEADER VARIABLES
PROGRAM_NUMBER="2."
NEW_REPORT= False


def load_finetuned_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    print("DEBUG: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("DEBUG: Tokenizer loaded successfully")
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=MODEL_PRECISION,
        device_map={"": 0},  # Force everything to first GPU
        # device_map="auto"
        # attn_implementation='eager'
    )
    print("DEBUG: Model loaded successfully")
    return tokenizer, model

def post_process_report(text):
    """Apply rule-based corrections to traffic reports"""
    
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

def generate_report(model, tokenizer, events, timestamp, max_length=512):
    """Generate a traffic report using the same format as training"""
    
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
        top_p=0.8,
        do_sample=True,
        repetition_penalty=1.2,
        num_return_sequences=1
    )
    
    # Decode and extract only the generated part
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # More robust extraction
    if "<start_of_turn>model\n" in generated_text:
        report_content = generated_text.split("<start_of_turn>model\n", 1)[1].strip()
    else:
        # Fallback
        report_content = generated_text[len(inference_prompt):].strip()
    
    # Apply your correction if needed
    corrected_content = post_process_report(report_content)
    
    # Create header
    header = f"Prometne informacije          {formatted_date}       {formatted_time}         {PROGRAM_NUMBER} program\n\nPodatki o prometu.\n\n"
    if NEW_REPORT:
        header = "NOVE " + header
    
    final_report = header + corrected_content
    
    return final_report, report_content


def test_model(model, tokenizer, test_data_path):
    """Test the model on traffic events data"""
    # Load test data with proper columns
    df = pd.read_csv(test_data_path)
    
  
    results = []
    
    for _, row in df.iterrows():
        timestamp = row['Datum']
        events = row['Input_porocilo']
        
        # Generate report with timestamp and program numbers
        (report,pre_check_report) = generate_report(model, tokenizer, events, timestamp, PROGRAM_NUMBER)
        results.append({
            "timestamp": timestamp,
            "events": events,
            "generated_report": report,
            "pre_check_report": pre_check_report,
            "target_report": row['Porocilo'] if 'Porocilo' in row else None
        })
    
    # Save results
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_file = os.path.join(TESTING_DIR, f"generated_reports_{job_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    """Main testing function"""
    # Create testing directory if it doesn't exist
    os.makedirs(TESTING_DIR, exist_ok=True)
    print(f"DEBUG: Created testing directory: {TESTING_DIR}")
    
    # Load model
    print("DEBUG: Loading model...")
    tokenizer, model = load_finetuned_model(PATH_TO_MODEL)
    print("DEBUG: Model loaded successfully")
    

    print(f"DEBUG: Using test data from: {PATH_TO_DATASET}")
    if not os.path.exists(PATH_TO_DATASET):
        print(f"ERROR: Test data file not found at {PATH_TO_DATASET}")
        return
    
    # Test the model
    print("DEBUG: Starting model testing...")
    results = test_model(model, tokenizer, PATH_TO_DATASET)
    print(f"DEBUG: Testing completed with {len(results)} results")
if __name__ == "__main__":
    main()