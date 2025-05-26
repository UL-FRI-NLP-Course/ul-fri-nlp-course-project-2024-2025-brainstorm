import os
import glob
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from datetime import datetime
from dateutil.parser import parse
import re

# API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=API_KEY)


def initialize_chat_session():
    model = genai.GenerativeModel('gemini-2.0-flash', 
                                generation_config=genai.GenerationConfig(
                                    temperature=0.3,
                                    top_p=0.9
                                ))
    chat = model.start_chat(history=[])
    
    # OCENJEVANJE PROMPT
    initial_prompt = """Radijska poročila vsebujejo informacije o različnih prometnih dogodkih in stanju na cestah. 
    Imamo generirana v slovenščini in pravilna poročila v slovenščini. Za posamezno prometno poročilo preveri sledeče metrike in jih oceni na skali od 0 do 10.
    1.a) za koliko procentov prometnih dogodkov je lokacija pravilna [0-10]
    1.b) za koliko procentov prometnih dogodkov je tip dogodka pravilen (npr zastoj, nesreca etc) [0-10]
    1.c) za koliko procentov prometnih dogodkov  je smer pravilna [0-10]
    1.d) koliko je v generiranem poročilu dogodkov, ki niso prisotni v pravilnem poročilu [0-10]

    Preveri metrike, ki se nanašajo na kvaliteto generiranega poročila kot so:
    2.a oceni jedernatost [0-10]
    2.b oceni slovnično pravilnost [0-10]
    2.c oceni stil [0-10]
    2.d oceni celotno kvaliteto [0-10]
    2.e oceni ali je poročilo primerno za radijsko poročilo [0-10]
    
    Proročaj jedernato. Povej le ime metrike in oceno. Ne dodajaj dodatnega besedila.
    """
    
    response = chat.send_message(initial_prompt)
    print(f"Chat session initialized with initial prompt.")
    print(f"Response: {response.text}")
    print("--------------------------------------------------")
    return chat

def evaluate_with_big_LLM(chat, timestamp, generated_report, true_report):
    print(f"Evaluating report for timestamp: {timestamp}")
    prompt = f"""Radijska poročila vsebujejo informacije o različnih prometnih dogodkih in stanju na cestah. 
    Imamo generirana v slovenščini in pravilna poročila v slovenščini. Za posamezno prometno poročilo preveri sledeče metrike in jih oceni na skali od 0 do 10.
    1.a) za koliko procentov prometnih dogodkov je lokacija pravilna [0-10] (utemeljitev ocene)
    1.b) za koliko procentov prometnih dogodkov je tip dogodka pravilen (npr zastoj, nesreca etc) [0-10] (utemeljitev ocene)
    1.c) za koliko procentov prometnih dogodkov  je smer pravilna [0-10] (utemeljitev ocene)
    1.d) koliko je v generiranem poročilu dogodkov, ki niso prisotni v pravilnem poročilu [0-10] (utemeljitev ocene)

    Preveri metrike, ki se nanašajo na kvaliteto generiranega poročila kot so:
    2.a oceni jedernatost [0-10] (utemeljitev ocene)
    2.b oceni slovnično pravilnost [0-10] (utemeljitev ocene)
    2.c oceni stil [0-10] (utemeljitev ocene)
    2.d oceni celotno kvaliteto [0-10] (utemeljitev ocene)
    2.e oceni ali je poročilo primerno za radijsko poročilo [0-10] (utemeljitev ocene)
    
    Proročaj jedernato. Povej le ime metrike in oceno. Ne dodajaj dodatnega besedila razen utemeljitve ocene v oklepajih.
    
    \n Generirano prometno poročilo: '{generated_report}'  \n  \n Pravilno prometno poročilo: '{true_report}' """
    
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        error_message = str(e)
        print(f"Error", error_message)
        return f"Error generating report: {e}"

def display_countdown(seconds):
    print(f"\nWaiting {seconds} seconds before processing next row...")
    start_time = time.time()
    end_time = start_time + seconds
    
    try:
        while time.time() < end_time:
            remaining = int(end_time - time.time())
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"\rCurrent time: {current_time} | Time remaining: {remaining} seconds until next API call...", end="")
            time.sleep(1)
        print("\nDelay complete. Continuing processing...")
    except KeyboardInterrupt:
        print("\nCountdown interrupted. Continuing...")
        return

def estimate_token_count(text, model_name='gemini-2.0-flash'):
    """Get accurate token count using Google's API."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.count_tokens(text)
        return response.total_tokens
    except Exception as e:
        # Fall back to estimation if API call fails
        print(f"Error counting tokens: {e}")
        return len(text) // 4  # Fallback to simple estimation

def process_csv_with_multiple_prompts(df_porocila, report_number="1"):
    try:
        chat_session = initialize_chat_session() 
        extracted_data = []
        error_before = 0
        
        for i, row in df_porocila.iterrows():
            row_datum = str(row['timestamp'])
            
            if not (pd.to_datetime(row_datum) > pd.to_datetime("2022-01-01 00:00:00")):
                continue
            
            print(row_datum)
            generated_report = row['generated_report']
            true_report = row['target_report']
            extracted_traff_events = evaluate_with_big_LLM(chat_session, row_datum, generated_report, true_report)
            
            if 'Error' in extracted_traff_events:
                # DELAY FOR EXCEEDING API
                display_countdown(69)
                extracted_traff_events = evaluate_with_big_LLM(chat_session, row_datum, generated_report, true_report)
                if 'Error' in extracted_traff_events: # EXCEEDED API FOR GOOD
                    break
                
            print(f"--- EVALUATION ---")
            print(extracted_traff_events[:1000] + "...")
            print("---------------------")
            
            extracted_data.append([row_datum, extracted_traff_events])
            
            df = pd.DataFrame(extracted_data, columns=['Datum', 'Prometni_dogodki'])
            df.to_csv('eval_w_gemini.csv', encoding='utf-8', index=False)
            
        df = pd.DataFrame(extracted_data, columns=['Datum', 'Prometni_dogodki'])
        df.to_csv(f'eval_w_gemini-{report_number}.csv', encoding='utf-8', index=False)
                
    except Exception as e:
        print(f"Error processing", e)
        
        
if __name__ == "__main__":
    print("Starting Gemini for generated traffic report evaluation.")

    FILE_NAME = "generated_reports_57976558.json"
    EXTRACTED_FILE_NUMBER = FILE_NAME.split("_")[-1].split(".")[0]
        
    df_results = pd.read_json(FILE_NAME)
    col_names = ['timestamp', 'events', 'generated_report', 'target_report']
        
    process_csv_with_multiple_prompts(df_results, EXTRACTED_FILE_NUMBER)