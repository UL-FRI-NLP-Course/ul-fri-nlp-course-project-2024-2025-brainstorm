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
PATH_TO_MODEL ="/d/hpc/projects/onj_fri/brainstorm/FineTune_models/GaMS-9B-Instruct/checkpoints/checkpoint-epoch-3"

# TEST DATASET PATH
PATH_TO_DATASET = "/d/hpc/projects/onj_fri/brainstorm/not_dataset_folder/not_test_data.csv"

# PROMPT FOR GENERATING REPORTS
PROMPT = """Generiraj prometno poročilo v slovenščini za radio.
        Upoštevaj standardno strukturo:
        1. Začni z "Prometne informacije [datum] [čas] za Radio Slovenija"
        2. Vključi samo pomembne prometne dogodke
        3. Uporabljaj trpne oblike in ustrezno terminologijo
        4. Poročilo naj bo jedrnato (<1 min branja)

        Podatki o prometu:
        """ 

# REPORT HEADER VARIABLES
PROGRAM_NUMBER="2."
NEW_REPORT= False


def load_finetuned_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=MODEL_PRECISION,
        device_map="auto",
        attn_implementation='eager'
    )
    
    return tokenizer, model

def correct_slovenian_text(text, model, tokenizer):
    """Use the model to correct grammar, style and formatting in Slovenian traffic reports"""
    
    correction_prompt = f"""Popravi naslednje prometno poročilo za radio.
    
    Navodila za popravke:
    1. Slovnična pravilnost: Popravi vse slovnične napake v slovenščini
    2. Trpnik: Uporabljaj trpno obliko (npr. "Zaprta je cesta" namesto "Cesta je zaprta")
    3. Prelomi odstavkov: Vsak dogodek naj bo v svojem odstavku, ločen z dvojnim presledkom
    4. Odstrani ponavljanja: Če se ista informacija pojavi večkrat, jo obdrži le enkrat
    5. Standardna terminologija: Uporabljaj standardne izraze za prometna poročila (npr. "oviran promet", "zastoj", "zaprta cesta")
    6. Končna ločila: Vsak odstavek naj se konča s piko, klicajem ali vprašajem
    7. Velike začetnice: Vsak odstavek naj se začne z veliko začetnico
    8. Poročilo ohrani jedrnato in primerno za radijsko branje
    9. Ohrani vse pomembne prometne informacije
    
    Poročilo za popravek:
    {text}
    
    Popravljeno poročilo:"""
    
    # Process with model
    inputs = tokenizer(correction_prompt, return_tensors="pt").to(model.device)
    
    # Generate corrected text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        temperature=0.3,  # Lower temperature for more deterministic corrections
        top_p=0.95,
        do_sample=True,
        num_return_sequences=1
    )
    
    # Decode the generated text
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the corrected portion after the prompt
    if "Popravljeno poročilo:" in corrected_text:
        corrected_text = corrected_text.split("Popravljeno poročilo:")[-1].strip()
    
    return corrected_text

def generate_report(model, tokenizer, events, timestamp, max_length=512):
    """Generate a traffic report based on events"""
    
    # Parse the timestamp to get formatted date and time
    date_obj = pd.to_datetime(timestamp)
    formatted_date = date_obj.strftime("%d. %m. %Y")
    formatted_time = date_obj.strftime("%H.%M")
    
    # Create the header
    header = f"Prometne informacije          {formatted_date}       {formatted_time}         {PROGRAM_NUMBER} program\n\nPodatki o prometu.\n\n"
    if NEW_REPORT:
        header= "NOVE "+header
    
    # Format input with appropriate prompt
    prompt = f"{PROMPT}\n{events}n\nReport:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1
    )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the report part (remove the prompt)
    report_content = generated_text[len(prompt):].strip()
    
    # NEW: Apply grammar and formatting correction

    corrected_content = correct_slovenian_text(report_content, model, tokenizer)
    
    # Add the header to the corrected content
    final_report = header + corrected_content
    
    return (final_report,report_content)

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