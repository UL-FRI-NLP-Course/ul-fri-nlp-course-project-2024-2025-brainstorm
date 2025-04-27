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

# Define the 5 different prompt approaches
PROMPT_APPROACHES = {
    "combined prompt": lambda example: f"""You are an employee at RTVSlo responsible for generating radio traffic information. Your task is to create concise, accurate, and stylistically correct traffic reports in Slovenian based on provided input data.

INPUT DATA HANDLING:
- Assume the input data is a list of traffic events, potentially unstructured or semi-structured (e.g., raw text reports, lists). Each event ideally includes type (accident, jam, obstacle, roadwork), location, direction, description, and a timestamp.
- Prioritize recency: If multiple reports exist for the same location, use the information from the report with the latest timestamp relevant to the report time ([date] [time]).
- Extract key information: Identify the type of incident, precise location (road name/number, section, kilometer marker if available), direction of travel, and consequences (lane closures, delay times, type of obstruction).
- Handle missing information: If specific details (like exact delay times) are missing in the input, report the situation generally (e.g., "Zastoj je daljši," "Promet je močno oviran") rather than inventing details. Do not report incidents where essential information like location or type is completely missing.

LANGUAGE CONSTRAINTS:
- Use only active voice constructions (e.g., "Zaprta je..." [It is closed], not "Pas je zaprt" [The lane is closed]).
- Always mention locations with the direction of travel (e.g., "proti Ljubljani" [towards Ljubljana], "proti Kopru" [towards Koper]).
- Use the precise date and time format: "DD. M. YYYY ob HH:MM" (e.g., "29. 6. 2024 ob 14:00").
- Avoid English expressions; use Slovenian equivalents (e.g., "zastoj" instead of "delay", "gneča" instead of "traffic jam" unless specifically referring to a long standstill as "zastoj").
- Use standard phrases: "oviran promet" (obstructed traffic), "nastaja zastoj" (traffic jam is forming), "promet poteka počasi" (traffic is slow).
- Include specific traffic status descriptions when available: "promet poteka po odstavnem pasu" (traffic flows on the emergency lane), "zaprta sta vozni in počasni pas" (driving and slow lanes are closed).
- Use "prometne informacije" (traffic information) for the report title, not "prometno poročilo" (traffic report).
- Include specific delay times when available and significant: "Čas potovanja se podaljša za približno 30 minut". Use standard time references: "predvidoma do 20. ure" (expected until 8 PM).

CONTENT FILTERING AND SIGNIFICANCE:
- Only include significant traffic incidents. Define "significant" as:
    - Accidents causing lane closures or delays estimated at 15 minutes or more, if not provided in data do not report.
    - Traffic jams resulting in delays estimated at 20 minutes or more, or those described as "daljši zastoj" (long jam) or "močan zastoj" (heavy jam) in the source data.
    - Obstacles completely blocking a lane or posing a direct safety hazard (e.g., large debris, stopped vehicle in a dangerous spot like a tunnel or before a bend).
    - Roadworks only if they are causing active, significant delays (approx. 15+ min) at the time of the report OR if they involve complete closures starting or ending today. Do not include routine, long-term roadworks unless they meet these criteria for the current reporting time.
    - If time estimates are not provided, use the following guidelines:
        - For accidents: Report if the situation is serious enough to cause significant delays or lane closures if provided in the data, if not do not report.
        - For traffic jams: Report if the situation is serious enough to cause significant delays or lane closures if provided in the data, if not do not report.
        - For obstacles: Report if the situation is serious enough to cause significant delays or lane closures if provided in the data, if not do not report.
        - For roadworks: Report if the situation is serious enough to cause significant delays or lane closures if provided in the dat), if not do not report.
- Filter based on report time: Exclude any incidents confirmed as resolved before the specified report time "[date] [time]". Ignore roadworks planned for the future or completed in the past relative to the report time.

STYLISTIC APPROACH AND FORMATTING:
- Use short, informative sentences focused on location and impact.
- Start the entire report directly with "Prometne informacije [date] [time] za 1., 2. in 3. program Radia Slovenija." (Fill in the actual date and time). No preceding text.
- Immediately following the header line, include the standalone line "Podatki o prometu.".
- Ensure a single blank line follows "Podatki o prometu." before the first traffic item.
- Group all reports for a specific road together (e.g., all traffic jams on the A1 Primorska highway appear consecutively). 
- Keep paragraphs concise, focusing on one specific event or a set of closely related events on the same road section (max 2-3 sentences typically).
- Strictly avoid markdown formatting, bullet points, numbered lists, HTML tags, or any visual separators in the final output. Use plain text only.
- Do not include category headings (like "Nesreče:", "Zastoji:") in the final output. The structure must be maintained implicitly by the order of the paragraphs.
- Use standard phrasing for reopened roads: "je spet prevozna" or "je spet odprta". Place significant reopening information at the end of the report, possibly in the final "Warnings/Restrictions/Reopenings" section.
- End paragraphs describing accidents or dangerous obstacles with "Opozarjamo na nevarnost naleta." where appropriate (i.e., where traffic is stopped or slowed abruptly). Do not add this warning if the incident is just a slow-down without stationary vehicles or if the hazard is minor.
- Maintain a neutral, factual, and clear tone appropriate for a public broadcast. Avoid sensationalism.
- IMPORTANT! Keep the report concise, ideally under 1 minute of reading time.
- Do not include ALL of the data in the report. Only include the most relevant and significant information. The goal is to provide a clear and concise summary of the traffic situation without overwhelming details.
- Group incidents together based on report structure order. All accidents are one paragraph, all traffic jams are another, etc. Do not mix them in the same paragraph. DO NOT WRITE EVERYTHING IN ONE LINE. Make it readable and easy to follow. For a radio report, it should be clear and easy to understand.
- Do not include any additional information or commentary outside the specified structure. The report should be purely factual and focused on the traffic situation.
- VERY IMPORTANT! - Avoid excessive repetition of events. If a road is closed due to an accident, do not repeat the same information in the Traffic jams or roadworks sections. Instead, summarize the situation in a single sentence. And do not report the same event multiple times in different sections (e.g., once as an accident and again as closed lane or traffic jam).


REPORT STRUCTURE (Implicit Order - No Headings in Output):
1. Header: "Prometne informacije [date] [time] za 1., 2. in 3. program Radia Slovenija."
2. Standalone Line: "Podatki o prometu."
3. Blank Line
4. Accidents (Significant accidents. Prioritize human impact/blockages.)
5. Traffic Jams (Only significant delays as defined above. Include obstacles causing delays.)
6. Roadworks (Only those causing significant delays today or full closures starting/ending today.)
7. Warnings / Restrictions / Reopenings (General warnings like weather impacts if severe, truck restrictions (e.g., weight/timing), information on major roads being reopened, buying of Vinjeta.)(Keep very short.)

EXAMPLES:
Here are examples demonstrating the desired structure and style:
{example}

EXECUTION:
Using the provided data, compose a report in Slovenian language that could be read on the radio, strictly following all the above instructions, structure, and constraints. Stick to the provided examples as closly as possible in terms of formatting. If the report is over 800 characters you will be fired as a RTVSlo reporter. Wait for the data input.""" ,
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

def display_countdown(seconds):
    """Display a countdown timer for the specified number of seconds."""
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

def process_csv_with_multiple_prompts():
    """Process the CSV file with multiple prompt approaches, grouping by date/time."""
    # Path to data folder
    data_dir = os.path.join(os.getcwd(), "..", "data_preprocessing", "Small_dataset")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at {data_dir}")
        print("Please add CSV file to this directory and run the script again.")
        return
    
    # Get all CSV files in the data folder
    csv_files = glob.glob(os.path.join(data_dir, "in.csv"))
    
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
        
        # Drop unnecessary columns to save on tokens but KEEP A1-C2
        columns_to_drop = ['LegacyId', 'Operater',
                          'TitleVremeSLO', 'ContentVremeSLO', 
                          'TitleMednarodneInformacijeSLO', 'ContentMednarodneInformacijeSLO', 
                          'TitleSplosnoSLO', 'ContentSplosnoSLO']
        
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Group by date (extract only date part, ignoring minor time differences)
        def extract_date_group(date_str):
            if pd.isna(date_str):
                return None
            try:
                date = parse(date_str)
                # Group by date and hour only (ignoring minutes)
                return f"{date.month}/{date.day}/{str(date.year)[2:]} {date.hour}"
            except:
                return None
        
        df['DateGroup'] = df['Datum'].apply(extract_date_group)
        
        # Create reports directory structure
        for prompt_type in PROMPT_APPROACHES.keys():
            os.makedirs(os.path.join("reports", prompt_type), exist_ok=True)

        # Read all .txt files in Small_dataset folder that contain out in their name
        txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
        out_files = [f for f in txt_files if "out" in os.path.basename(f)]

        if not out_files:
            print("No example output files found in the data directory.")
            return
        
        # Read all example outputs and store them together with \n\n as separator into example_output
        example_output = ""

        for out_file in out_files:
            with open(out_file, "r", encoding="utf-8") as f:
                example_output += f.read() + "\n\n"


        print(f"Example output files found: {len(out_files)}")
        print(f"Example output files content:\n{example_output}")

                
        # Read example output file
        # example_path = os.path.join(os.getcwd(), "..", "data_preprocessing", "Small_dataset", "out1.txt")
        # print(f"Example output file path: {example_path}")
        # if not os.path.exists(example_path):
        #     print(f"Error: Example output file {example_path} not found")
        #     return
            
        # with open(example_path, "r", encoding="utf-8") as f:
        #     example_output = f.read().strip()
        
        # Process each date group
        date_groups = df['DateGroup'].unique()
        total_token_count = 0
        
        for i, date_group in enumerate(date_groups):
            if pd.isna(date_group):
                continue
                
            print(f"\nProcessing date group: {date_group}")
            
            # Get all rows for this date group
            group_df = df[df['DateGroup'] == date_group].copy()
            
            # Extract time range information for report header
            min_time = "Unknown"
            max_time = "Unknown"
            report_hour = "Unknown"
            
            try:
                # Get the earliest and latest times in this group
                if 'Datum' in group_df.columns and not group_df['Datum'].empty:
                    times = [parse(d) for d in group_df['Datum'].dropna() if pd.notna(d)]
                    if times:
                        min_time = min(times).strftime("%H:%M")
                        max_time = max(times).strftime("%H:%M")
                        # Report hour is the next hour (rounded up)
                        next_hour = max(times).replace(minute=0, second=0, microsecond=0)
                        if max(times).minute > 0 or max(times).second > 0:
                            next_hour = next_hour.replace(hour=next_hour.hour + 1)
                        report_hour = next_hour.strftime("%H:00")
            except Exception as e:
                print(f"Error extracting time range: {e}")
            
            # Add a delay between date groups (except for the first group)
            if i > 0:
                display_countdown(70)
            
            # Merge content from all rows in the group, removing duplicates
            merged_data = {}
            
            # Content columns to merge - include A1-C2
            content_columns = [
                'A1', 'B1', 'C1', 'A2', 'B2', 'C2',
                'TitlePomembnoSLO', 'ContentPomembnoSLO',
                'TitleNesreceSLO', 'ContentNesreceSLO',
                'TitleZastojiSLO', 'ContentZastojiSLO',
                'TitleOvireSLO', 'ContentOvireSLO',
                'TitleDeloNaCestiSLO', 'ContentDeloNaCestiSLO',
                'TitleOpozorilaSLO', 'ContentOpozorilaSLO'
            ]
            
            # Improved handling of duplicates - keep all unique content
            for col in content_columns:
                if col in group_df.columns:
                    # Get all non-null values
                    values = [v for v in group_df[col].dropna().tolist() if str(v).upper() != "NULL" and str(v).strip() != ""]
                    
                    # Remove actual duplicates while preserving order
                    unique_values = []
                    seen = set()
                    for value in values:
                        # Clean HTML tags and normalize whitespace for comparison
                        cleaned = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', str(value))).strip()
                        if cleaned not in seen and cleaned:
                            seen.add(cleaned)
                            unique_values.append(value)
                    
                    if unique_values:
                        merged_data[col] = unique_values
            
            # Convert merged data to a clean string format
            clean_data = []
            
            # Add time range information at the beginning
            time_info = f"REPORT_TIME: Podatki zajeti od {min_time} do {max_time}, poročilo za {report_hour}"
            clean_data.append(time_info)
            
            for col, values in merged_data.items():
                # Include all unique values for each column
                for value in values:
                    # Clean HTML tags
                    value = re.sub(r'<[^>]+>', ' ', str(value))
                    # Remove excessive whitespace
                    value = re.sub(r'\s+', ' ', value).strip()
                    if value:  # Skip empty strings
                        clean_data.append(f"{col}: {value}")
            
            row_data = "\n\n".join(clean_data)
            
            # Estimate token count and display
            token_estimate = estimate_token_count(row_data)
            total_token_count += token_estimate
            print(f"\n=== TOKEN ESTIMATION ===")
            print(f"Group: {date_group}")
            print(f"Report time: {report_hour}")
            print(f"Data time range: {min_time} to {max_time}")
            print(f"Text length: {len(row_data)} characters")
            print(f"Token count: {token_estimate} tokens (via Google's API)")
            print(f"======================")
            
            # Save the processed data for reference
            data_file = os.path.join("reports", f"{date_group.replace('/', '-').replace(' ', '_')}_processed_data.txt")
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(row_data)
            print(f"Processed data saved to {data_file}")
            
            # Initialize chat sessions for all prompt approaches
            chat_sessions = {}
            for prompt_type in PROMPT_APPROACHES.keys():
                chat_sessions[prompt_type] = initialize_chat_session(prompt_type, example_output)
                
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
                output_file = os.path.join("reports", prompt_type, f"{date_group.replace('/', '-').replace(' ', '_')}_report.txt")
                
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                
                print(f"Report saved to {output_file}")
        
        print(f"\n### TOTAL ESTIMATED TOKENS USED: {total_token_count} ###")
            
    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    print("Starting Gemini Traffic Reporter with Multiple Prompt Approaches")
    process_csv_with_multiple_prompts()