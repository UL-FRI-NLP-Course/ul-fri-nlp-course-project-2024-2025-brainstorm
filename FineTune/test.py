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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{TESTING_DIR}/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_finetuned_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    return tokenizer, model

def generate_report(model, tokenizer, events, max_length=512):
    """Generate a traffic report based on events"""
    prompt = f"Generate a traffic report based on the following events:\n{events}\n\nReport:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the report part (remove the prompt)
    report = generated_text[len(prompt):]
    
    return report.strip()

def test_model(model, tokenizer, test_data_path, num_samples=5):
    """Test the model on traffic events data"""
    # Load test data with proper columns
    df = pd.read_csv(test_data_path)
    
    # Take a sample of events to generate reports for
    if len(df) > num_samples:
        df = df.sample(num_samples, random_state=42)
    
    results = []
    
    for _, row in df.iterrows():
        timestamp = row['Datum']
        events = row['Input_porocilo']
        
        logger.info(f"Generating report for {timestamp}")
        logger.info(f"Events: {events}")
        
        # Generate report
        report = generate_report(model, tokenizer, events)
        
        # If we have the target report, include it for comparison
        if 'Porocilo' in row:
            logger.info(f"Target report: {row['Porocilo']}")
            logger.info(f"Generated report: {report}")
        else:
            logger.info(f"Generated report: {report}")
            
        logger.info("-" * 80)
        
        results.append({
            "timestamp": timestamp,
            "events": events,
            "generated_report": report,
            "target_report": row['Porocilo'] if 'Porocilo' in row else None
        })
    
    # Save results
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    output_file = os.path.join(TESTING_DIR, f"generated_reports_{job_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved test results to {output_file}")
    
    return results

def main():
    """Main testing function"""
    # Debug outputs
    print(f"DEBUG: Current directory: {os.getcwd()}")
    print(f"DEBUG: TESTING_DIR set to: {TESTING_DIR}")
    print(f"DEBUG: OUTPUT_DIR set to: {OUTPUT_DIR}")
    print(f"DEBUG: DATASET_DIR set to: {DARASET_DIR}")
    
    # Create testing directory if it doesn't exist
    os.makedirs(TESTING_DIR, exist_ok=True)
    print(f"DEBUG: Created testing directory: {TESTING_DIR}")
    
    # Load model
    print("DEBUG: Loading model...")
    tokenizer, model = load_finetuned_model(OUTPUT_DIR)
    print("DEBUG: Model loaded successfully")
    
    # Test data path
    test_data_path = DARASET_DIR
    print(f"DEBUG: Using test data from: {test_data_path}")
    
    if not os.path.exists(test_data_path):
        print(f"ERROR: Test data file not found at {test_data_path}")
        return
    
    # Test the model
    print("DEBUG: Starting model testing...")
    results = test_model(model, tokenizer, test_data_path)
    print(f"DEBUG: Testing completed with {len(results)} results")
if __name__ == "__main__":
    main()