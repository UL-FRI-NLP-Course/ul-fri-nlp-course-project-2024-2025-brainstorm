import os
import glob
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables and API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure Gemini API
genai.configure(api_key=API_KEY)

def initialize_chat_session(example_output):
    """Initialize a chat session with Gemini and provide an example output format."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    chat = model.start_chat(history=[])
    
    # First message to establish context and provide example output
    initial_prompt = f"Si zaposlen na RTVSlo in odgovoren za prometna poročila. Poročila so opremljena z informacijo o datumu in vsebujejo informacije o prometu brez podnaslovov, strnjena v enotno, jasno razumljivo prometno poročilo. Podatke brez vsebinskih naslovov poveži v smiselno enoto. Navedi jih po pomembnosti: nesreče na začetku, ovire in zastoji potem, splošna obvestila in na koncu morebitne prodaje ali vremensko stanje, če vpliva direktno na promet (dež, sneg, toča). Tole je primer izpisa, počakaj na podatke: \n\n{example_output}. "
    
    response = chat.send_message(initial_prompt)
    print("Chat session initialized with example output format.")
    
    return chat

def generate_traffic_report(chat, row_data):
    """Generate a traffic report using an ongoing chat session."""
    prompt = f"Sestavi prometno poročilo iz teh podatkov: \n\n{row_data}"
    response = chat.send_message(prompt)
    return response.text

def process_csv_file():
    """Process the CSV file with a contextual approach."""
    # Path to data folder
    # Move one dir out and go to data_preprocessed folder
    # This is the folder where the CSV files are expected to be found
    data_dir = os.path.join(os.getcwd(), "..", "data_preprocessing", "Small_dataset")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")
        print("Please add CSV file to this directory and run the script again.")
        return
    
    # Get all CSV files in the data folder
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the data directory.")
        return
    
    # Use the first CSV file found
    file_path = csv_files[0]
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}")
    
    try:
        # Read CSV file
        df = pd.read_csv(file_path, encoding="utf-8", sep=";")
        
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        
        # Read example output file (just using out1.txt as a general example)
        example_path = os.path.join(os.getcwd(), "..", "data_preprocessing", "Small_dataset", "out1.txt")
        print(f"Example output file path: {example_path}")
        if not os.path.exists(example_path):
            print(f"Error: Example output file {example_path} not found")
            return
            
        with open(example_path, "r", encoding="utf-8") as f:
            example_output = f.read().strip()
        
        # Initialize chat session with the example output
        chat = initialize_chat_session(example_output)
        
        # Process each row in the CSV
        for index, row in df.iterrows():
            # Skip header row if present
            if index == 0 and "header" in str(row).lower():
                continue
                
            print(f"\nProcessing row {index+1}")
            
            # Convert row data to string
            row_data = row.to_string()
            
            # Generate traffic report using the ongoing chat session
            report = generate_traffic_report(chat, row_data)
            
            # Print report
            print("\n--- Traffic Report ---")
            print(report)
            print("---------------------\n")
            
            # Save report to output file
            output_file = os.path.join("reports", f"row{index+1}_report.txt")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Traffic Report for row {index+1}\n\n")
                f.write(report)
            
            print(f"Report saved to {output_file}")
            
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Gemini Traffic Reporter")
    process_csv_file()