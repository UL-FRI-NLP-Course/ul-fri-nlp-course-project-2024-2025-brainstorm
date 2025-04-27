import os
import glob
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures

# Load environment variables and API key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# Configure Gemini API
genai.configure(api_key=API_KEY)

# Define the 5 different prompt approaches
PROMPT_APPROACHES = {
    "template_enforcement": lambda example: f"""Si zaposlen na RTVSlo in odgovoren za prometna poročila. 
Struktura poročila MORA biti:
[Datumska glava] 
[Nesreče] (najpomembnejše, vsaka v lastni kratki alineji)
[Zastoji] (glavne lokacije)
[Ovire] 
[Delo na cesti] 
[Opozorila] (vključno s tovornjaki)
Brez podnaslovov. Uporabi naravno slovenski jezik brez robotaškega tona. Primer strukture: 

{example}

Iz danih podatkov sestavi poročilo po zgornji strukturi. Počakaj na podatke.""",

    "style_mimicry": lambda example: f"""Si zaposlen na RTVSlo in odgovoren za prometna poročila.
Posnemi natančno stil primerjave: 
- Uporabi kratke, informativne stavke z lokacijami v krepkem
- Začni neposredno z "Prometne informacije [datum] [ura]"
- Združuj podobne incidente (npr. "na primorski avtocesti..." pokrij vse primorske primere skupaj)
- Izbeguj markdown, uporabi samo ** za poudarke
Tvoj izhod naj zgleda kot neposreden nadaljevanje tega primera: 

{example}

Počakaj na podatke.""",

    "content_prioritization": lambda example: f"""Si zaposlen na RTVSlo in odgovoren za prometna poročila.
Hierarhija vsebine:
1. Najprej navedi VSE nesreče z natančnimi lokacijami in posledicami
2. Nato zastoji po pomembnosti (avtoceste > regionalne ceste)
3. Dela na cestah kronološko (trenutna > prihodnja)
4. Opozorila samo če direktno vplivajo na promet
Formatiraj kot tekstočni poročevalec RTVSlo - brez številk, seznamov ali HTML. Primer pravilne ureditve: 

{example}

Počakaj na podatke.""",

    "linguistic_constraints": lambda example: f"""Si zaposlen na RTVSlo in odgovoren za prometna poročila.
Jezične omejitve:
- Uporabi samo aktivne konstrukcije ("Zaprta je...", ne "Pas je zaprt")
- Lokacije vedno navedi s smerjo gibanja ("proti Ljubljani")
- Za čas uporabi format "29. 6. 2024 ob 14:00" (ne "14:00 29/6/24")
- Izbegaj angleške izraze ("zastoj" ne "delay")
Prikaži informacije iz vhodnih podatkov v tem jezikovnem okviru. Primer: 

{example}

Počakaj na podatke.""",

    "structured_input_output": lambda example: f"""Si zaposlen na RTVSlo in odgovoren za pripravo prometnih poročil.  
- IZ PODATKOV izlušči ključne kategorije:  
  1) Nesreče, 2) Zastoji, 3) Ovire, 4) Delo na cesti, 5) Opozorila, 6) Tovorna vozila, 7) Vreme/Drugo.  
- Poročilo oblikuj kot tekočo, humanizirano novico:  
  • Odpri z datumom in uro,  
  • "Nesreče:" postavi na začetek,  
  • nato "Zastoji:", itd.,  
  • vsako kategorijo zapiši v enem odstavku brez dodatnih naslovkov.
  
Primer:
{example}

Počakaj na podatke."""
}

def initialize_chat_session(prompt_type, example_output):
    """Initialize a chat session with Gemini and provide an example output format."""
    model = genai.GenerativeModel('gemini-2.0-flash', 
                                generation_config=genai.GenerationConfig(
                                    temperature=0.3,
                                    top_p=0.9
                                ))
    chat = model.start_chat(history=[])
    
    # Get the appropriate prompt template based on the prompt_type
    initial_prompt = PROMPT_APPROACHES[prompt_type](example_output)
    
    response = chat.send_message(initial_prompt)
    print(f"Chat session initialized with {prompt_type} prompt.")
    
    return chat

def generate_traffic_report(chat, row_data, prompt_type):
    """Generate a traffic report using an ongoing chat session."""
    prompt = f"""Sestavi prometno poročilo iz teh podatkov, vendar uporabi svoje besede in izogibaj se dobesednemu kopiranju izrazov. 
    
    Preoblikuj in prepiši podatke na izviren način, ki ohranja pomen, a uporablja alternativne izraze in različne stavčne strukture: \n\n{row_data}"""
    
    try:
        response = chat.send_message(prompt)
        return response.text
    except Exception as e:
        error_message = str(e)
        print(f"Error with {prompt_type}: {error_message}")
        
        # If recitation error occurs, retry with higher temperature
        if "RECITATION" in error_message:
            try:
                print(f"Retrying with higher temperature...")
                model = genai.GenerativeModel('gemini-2.0-flash', 
                                          generation_config=genai.GenerationConfig(
                                              temperature=0.7,  # Increased temperature
                                              top_p=0.95        # Slightly higher top_p
                                          ))
                
                # Create a new chat session for the retry
                chat = model.start_chat(history=[])
                
                # Use a modified prompt that encourages paraphrasing
                retry_prompt = f"""Kot uslužbenec RTVSLO, sestavi izvirno prometno poročilo iz teh podatkov:
                
                {row_data}
                
                POMEMBNO: Ne kopiraj obstoječih fraz. Popolnoma preoblikuj vsebino z izvirnimi besedami, a ohrani pomen.
                Vsak stavek napiši drugače, kot bi bil zapisan v uradnih obvestilih."""
                
                response = chat.send_message(retry_prompt)
                return "⚠️ [Uporabljeno parafraziranje zaradi varnostnih omejitev]\n\n" + response.text
            except Exception as retry_error:
                return f"Error generating report: {retry_error}"
        else:
            return f"Error generating report: {e}"

def process_csv_with_multiple_prompts():
    """Process the CSV file with multiple prompt approaches."""
    # Path to data folder
    data_dir = os.path.join(os.getcwd(), "..", "data_preprocessing", "Small_dataset")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")
        print("Please add CSV file to this directory and run the script again.")
        return
    
    # Get all CSV files in the data folder
    csv_files = glob.glob(os.path.join(data_dir, "in_tmp.csv"))
    
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
        
        # Create reports directory structure
        for prompt_type in PROMPT_APPROACHES.keys():
            os.makedirs(os.path.join("reports", prompt_type), exist_ok=True)
        
        # Read example output file
        example_path = os.path.join(os.getcwd(), "..", "data_preprocessing", "Small_dataset", "out1.txt")
        print(f"Example output file path: {example_path}")
        if not os.path.exists(example_path):
            print(f"Error: Example output file {example_path} not found")
            return
            
        with open(example_path, "r", encoding="utf-8") as f:
            example_output = f.read().strip()
        
        # Initialize chat sessions for each prompt approach
        chat_sessions = {}
        for prompt_type in PROMPT_APPROACHES.keys():
            chat_sessions[prompt_type] = initialize_chat_session(prompt_type, example_output)
        
        # Process each row in the CSV
        for index, row in df.iterrows():
            # Skip header row if present
            if index == 0 and "header" in str(row).lower():
                continue
                
            print(f"\nProcessing row {index+1} with all prompt approaches")
            
            # Convert row data to a cleaner string format
            row_data = "\n".join([f"{col}: {value}" for col, value in row.items() 
                                 if pd.notna(value) and str(value).upper() != "NULL"])
            
            # Generate reports using all prompt approaches
            for prompt_type, chat in chat_sessions.items():
                print(f"\nGenerating report with {prompt_type} approach...")
                
                # Generate traffic report
                report = generate_traffic_report(chat, row_data, prompt_type)
                
                # Print summary
                print(f"--- {prompt_type} Traffic Report (first 100 chars) ---")
                print(report[:100] + "...")
                print("---------------------")
                
                # Save report to output file
                output_file = os.path.join("reports", prompt_type, f"row{index+1}_report.txt")
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                
                print(f"Report saved to {output_file}")
            
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Gemini Traffic Reporter with Multiple Prompt Approaches")
    process_csv_with_multiple_prompts()