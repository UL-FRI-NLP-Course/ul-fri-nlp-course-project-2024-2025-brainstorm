# filepath: /d/hpc/projects/onj_fri/brainstorm/news_generation.py
print("--- Python script started ---", flush=True)
import os
import torch
print("--- Imported torch ---", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer
print("--- Imported transformers ---", flush=True)
import pandas as pd
print("--- Imported pandas ---", flush=True)
import traceback # Make sure traceback is imported
print("--- Imported traceback ---", flush=True)
from tqdm import tqdm
print("--- Imported tqdm ---", flush=True)

# Define model path and input file path
model_path="/d/hpc/projects/onj_fri/brainstorm/models/Llama-4-Maverick-17B-128E-Instruct"
input_csv_path = "/d/hpc/projects/onj_fri/brainstorm/Small_dataset/in.csv"
print(f"--- Model path: {model_path} ---", flush=True)
print(f"--- Input CSV path: {input_csv_path} ---", flush=True)
# It's recommended to have an example output file or structure to guide the prompt better.
# Assuming 'out1' contains example news reports. We'll use a generic prompt structure for now.
# example_output_path = "out1" # If you have an example output file
from bs4 import BeautifulSoup
import re # Import regex for potentially cleaner text extraction
print("--- Imported BeautifulSoup and regex ---", flush=True)

def generate_report(model, tokenizer, traffic_info_dict, device):
    """Generates a traffic news report using the provided model and traffic info."""
    # --- Prompt Construction ---
    # --- Pre-process Input Data (Extract Text from HTML) ---
    processed_details = {}
    for key, value in traffic_info_dict.items():
        if isinstance(value, str) and ('<' in value and '>' in value): # Basic check for HTML
            soup = BeautifulSoup(value, 'lxml')
            # Get text, remove extra whitespace, handle None
            text = soup.get_text(separator=' ', strip=True)
            # Optional: Remove URLs that might be left over
            text = re.sub(r'http\S+', '', text)
            processed_details[key] = text if text else "N/A" # Use "N/A" if extraction yields nothing
        elif pd.isna(value): # Handle NaN values explicitly
             processed_details[key] = "N/A"
        else:
             processed_details[key] = value # Keep non-HTML, non-NaN values as is

    # --- Prompt Construction ---
    # Use the processed text details
    traffic_details_text = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in processed_details.items() if value != "N/A"]) # Only include non-N/A values

    # Construct the prompt with cleaned text
    prompt = f"""Generate a concise traffic news report based on the following data points:
{traffic_details_text}

Format the report as a brief news update suitable for a radio broadcast. Focus on the key information like location, event type, and impact."""
    # print(f"DEBUG: Using Prompt:\n{prompt}\n") # Optional: uncomment to see the cleaned prompt

    # --- Text Generation ---
    # Ensure the input tensors are created on the correct device implicitly via model dispatch
    inputs = tokenizer(prompt, return_tensors="pt") # .to(device) is not needed when model uses device_map

    # Adjust generation parameters as needed
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Post-processing ---
    # Try to remove the prompt from the beginning of the generated text
    if generated_text.startswith(prompt):
         report = generated_text[len(prompt):].strip()
    else:
         # Fallback if prompt isn't exactly at the start (might happen)
         # This part might need refinement based on actual model output
         report = generated_text.replace(prompt, "").strip() # Less reliable fallback

    if not report:
        report = "Model did not generate a distinct report."

    return report

if __name__ == "__main__":
    print("Starting Llama model traffic news generation...", flush=True)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available, running on CPU. This will be very slow.")

    try:
        # --- Load Model and Tokenizer ---
        print(f"Loading tokenizer from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"Loading model from {model_path} using device_map...", flush=True) # Updated print statement
        model_dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
        max_memory_map = {0: "60GiB", 1: "60GiB"} # Allocate ~35GiB to GPU 0 and GPU 1
        # Load the model using device_map="auto" (Requires accelerate)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            device_map="auto",     # Keep this - it's working!
            max_memory=max_memory_map, # Use the updated map for 2 GPUs
            
            trust_remote_code=True
        )
        
        # Explicitly move the entire model to the designated device (GPU)
        # print(f"Moving model to device: {device}...")
        # model.to(device)
        print("Model and tokenizer loaded successfully.", flush=True)

        # --- Read Input Data ---
        print(f"Reading input data from {input_csv_path}...", flush=True)
        try:
            # Specify the delimiter and ensure the header is read correctly
            df = pd.read_csv(input_csv_path, delimiter=';', header=0)
            print("Input data read successfully.")
            print(f"Found columns: {df.columns.tolist()}")
            print(f"Number of records: {len(df)}")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_csv_path}")
            exit(1)
        except Exception as e:
            print(f"Error reading CSV file '{input_csv_path}': {e}")
            exit(1)

        # --- Generate Reports ---
        print("\n--- Generating Reports ---", flush=True)
        all_reports = []
        for index, row in df.iterrows():
            print(f"\nProcessing record {index + 1}/{len(df)}...")
            # Convert the current row to a dictionary
            traffic_info_dict = row.to_dict()
            print(f"Input data: {traffic_info_dict}")

            # Generate the report
            try:
                report = generate_report(model, tokenizer, traffic_info_dict, device)
                print("\nGenerated Report:")
                print(report)
                all_reports.append(report)
            except Exception as e:
                print(f"Error generating report for record {index + 1}: {e}")
                all_reports.append(f"Error: Could not generate report for record {index + 1}")
            print("-" * 25)

        # --- Optional: Save Reports ---
        # You can save the generated reports to a file if needed
        output_file = "/d/hpc/projects/onj_fri/brainstorm/generated_reports/generated_reports.txt"
        print(f"\nSaving reports to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as f:
            for i, report in enumerate(all_reports):
                f.write(f"--- Report for Record {i+1} ---\n")
                f.write(report + "\n\n")
        print("Reports saved.")

        print("\nFinished generating reports.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging