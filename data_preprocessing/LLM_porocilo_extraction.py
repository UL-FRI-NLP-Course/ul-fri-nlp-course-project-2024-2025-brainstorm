import os
import glob
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from datetime import datetime
from dateutil.parser import parse
import re

# Load environment variables and API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure Gemini API
genai.configure(api_key=API_KEY)


def initialize_chat_session():
    """Initialize a chat session with Gemini and provide an example output format."""
    
    model = genai.GenerativeModel('gemini-2.0-flash', 
                                generation_config=genai.GenerationConfig(
                                    temperature=0.3,
                                    top_p=0.9
                                ))
    chat = model.start_chat(history=[])
    
    # TUKAJ V INITAL PROMT lahk dodamo se domain knowledge o cestah 
    initial_prompt = "Radijska poročila vsebujejo informacije o različnih prometnih dogodkih in stanju na cestah. Za analizo stanja na cesti je ključno, da izločimo podatke o prometnih dogodkih iz posameznega prometnega poročila."
    
    response = chat.send_message(initial_prompt)
    print(f"Chat session initialized with initial prompt.")
    
    return chat

def extract_data_from_porocilo(chat, row_data):
    """Generate a traffic report using an ongoing chat session."""
    
    prompt = f"""Iz prometnega poročila pridobi informacije o posameznih prometnih dogodkih. Poročaj o lokaciji, smeri in tipu prometnega dogodka v obliki: LOKACIJA: lokacija | SMER: smer | TIP: tip dogodka. V odgovor vključi le te informacije brez dodatnega besedila. \n Prometno poročilo: '{row_data}'"""
    
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        error_message = str(e)
        print(f"Error", error_message)
        
        # If recitation error occurs, retry with higher temperature
        if "RECITATION" in error_message:
            try:
                print(f"Retrying with higher temperature...")
                model = genai.GenerativeModel('gemini-2.0-flash', 
                                          generation_config=genai.GenerationConfig(
                                              temperature=0.7,
                                              top_p=0.95       
                                          ))
                
                chat = model.start_chat(history=[])
                retry_prompt = f"""Izlušči informacije o prometnih dogodkih, njihovi lokaciji, smeri in tipu dogodka za prometno radijsko poročilo:
                
                {row_data}
                ."""
                
                response = chat.send_message(retry_prompt)
                return "⚠️ [Uporabljeno parafraziranje zaradi varnostnih omejitev]\n\n" + response.text
            except Exception as retry_error:
                return f"Error generating report: {retry_error}"
        else:
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

def process_csv_with_multiple_prompts(df_porocila):
    try:
        chat_session = initialize_chat_session()

        print(f"\nExtracting data")
        
        extracted_data = []
        error_before = 0
        
        for i, row in df_porocila.iterrows():
            row_datum = str(row['Datum'])
            
            if not (pd.to_datetime(row_datum) > pd.to_datetime("2022-01-21 00:00:00")):
                continue
            
            print(row_datum)
            row_porocilo = row['Porocilo']
            extracted_traff_events = extract_data_from_porocilo(chat_session, row_porocilo)
            
            if 'Error' in extracted_traff_events:
                # DELAY FOR EXCEEDING API
                display_countdown(69)
                extracted_traff_events = extract_data_from_porocilo(chat_session, row_porocilo)
                if 'Error' in extracted_traff_events: # EXCEEDED API FOR GOOD
                    break
                
            print(f"--- Traffic events (first 500 chars) ---")
            print(extracted_traff_events[:500] + "...")
            print("---------------------")
            
            extracted_data.append([row_datum, extracted_traff_events])
            
            df = pd.DataFrame(extracted_data, columns=['Datum', 'Prometni_dogodki'])
            df.to_csv('prometni_dogodki_llm2.csv', encoding='utf-8', index=False)
            
        df = pd.DataFrame(extracted_data, columns=['Datum', 'Prometni_dogodki'])
        df.to_csv('prometni_dogodki_llm2.csv', encoding='utf-8', index=False)
                
    except Exception as e:
        print(f"Error processing", e)
        
if __name__ == "__main__":
    print("Starting Gemini traffic event extraction from the reports.")
    df_input = pd.read_csv('porocila_input_agg_prompt.csv', encoding='utf-8-sig', sep=';')
    df_output = pd.read_csv('porocila_data.csv', encoding='utf-8-sig', sep=';')

    df_input['Datum'] = pd.to_datetime(df_input['Datum'])
    df_output['Datum'] = pd.to_datetime(df_output['Datum'])
    process_csv_with_multiple_prompts(df_output)