import numpy as np 
import pandas as pd 
import re, os

import multiprocessing
from functools import partial
from tqdm import tqdm

import nltk
from nltk.collocations import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

from datetime import timedelta, datetime
import csv
import keyboard

from colorama import Fore
from colorama import init
from concurrent.futures import ThreadPoolExecutor  # Add this at the top of your file

init()

# INPUT AND OUTPUT FILES =============================================================

df_input = pd.read_csv('porocila_input_agg_prompt.csv', encoding='utf-8-sig', sep=';')
df_output = pd.read_csv('porocila_data.csv', encoding='utf-8-sig', sep=';')
df_traffic_events = pd.read_csv('prometni_dogodki_llm.csv', encoding='utf-8-sig', sep=',')

df_input['Datum'] = pd.to_datetime(df_input['Datum'])
df_output['Datum'] = pd.to_datetime(df_output['Datum'])
df_traffic_events['Datum'] = pd.to_datetime(df_traffic_events['Datum'])

df_input = df_input.sort_values('Datum')
df_output = df_output.sort_values('Datum')
df_traffic_events = df_traffic_events.sort_values('Datum')

start_datum = pd.Timestamp(2022, 1, 1, 10)
df_output = df_output[df_output['Datum'] > start_datum]
df_output = df_output[df_output['Datum'].dt.hour.between(8, 18)]

data_dir = 'RTVSlo/RTVSlo'
data_dir = '.'
file_path = os.path.join(data_dir, 'podatki_input.xlsx')
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names
df_input_excel = pd.concat([excel_file.parse(sheet) for sheet in sheet_names], ignore_index=True)

df_input_excel['Datum'] = pd.to_datetime(df_input_excel['Datum'])
df_input_excel = df_input_excel.sort_values('Datum')

#df_input_excel.drop(columns=[['LegacyId', 'Operater', 'A1', 'B1']], inplace=True, errors='ignore')
#df_input_excel = df_input_excel[['Datum', 'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentOpozorilaSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentVremeSLO', 'ContentSplosnoSLO']]
content_columns_datum = ['Datum', 'A1', 'B1',
        'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentOpozorilaSLO',
        'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentVremeSLO', 
        'ContentSplosnoSLO'
    ]
df_input_excel = df_input_excel[content_columns_datum]
df_input_excel['Datum'] = pd.to_datetime(df_input_excel['Datum'])

# SIMILARITY FUNCTIONS ========================================================

def get_embedding(text, tokenizer, model, device=None):
    if device is None:
        device = next(model.parameters()).device
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    
    # Move result back to CPU for numpy compatibility
    return mean_pooled.squeeze().cpu().numpy()

# ================================================================================

def preprocess_html(text):
    arr = re.split(r'</?p>', text)
    return [el.strip() for el in arr if el.strip()]

def get_most_similar_content(df_input, rows_indices, query, model, tokenizer, device=None):   
    content_columns = ['A1', 'B1',
        'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentOpozorilaSLO',
        'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentVremeSLO', 
        'ContentSplosnoSLO']
    
    if device is None:
        device = next(model.parameters()).device
        
    documents = []
    # Collection phase - same as before
    for idx in rows_indices:
        row = df_input.iloc[idx]
        for col in content_columns:
            text_content = row[col]
            
            if col in ['A1', 'B1']:
                if isinstance(text_content, (int, float)) and np.isnan(text_content):
                    continue
                posamezno = preprocess_html(text_content)
                for tema in posamezno:
                    documents.append(tema)
            else:
                col_name = col.replace('Content', '')
                col_name = col_name.replace('SLO', '')
                if isinstance(text_content, str):  
                    documents.append(col_name + ': ' + text_content)
    
    # BATCH EMBEDDING - this is the key optimization
    # Process in batches of 32 (or adjust based on your GPU memory)
    batch_size = 128
    document_embeddings = []
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        # Tokenize as a batch
        inputs = tokenizer(batch_docs, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        document_embeddings.extend(batch_embeddings)
    
    # Get query embedding
    query_embedding = get_embedding(query, tokenizer, model, device)
    
    # Calculate similarities
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # Find best matches - same as before
    top_k = 3
    top_indices = similarities.argsort()[-top_k:][::-1]
    best_match = documents[top_indices[0]]
    top_sim = similarities[top_indices[0]]
    # print(Fore.BLUE + f"Score: {top_sim:.4f}:\nMatch: {best_match}\n")
    return best_match, top_sim

                
def retrieve_surrounding(selected_timestamp, df_input, traffic_dogodki, window_size, model, tokenizer):
    df_prior = df_input[df_input['Datum'] < selected_timestamp]
    rows_within_window = df_prior.tail(window_size)
    
    if rows_within_window.empty:
        print(Fore.RED + "No or incorrect input data.")
        return False, ""

    rows_indices = rows_within_window.index.tolist()
    events = traffic_dogodki.split('\n')
    best_match_content_all = ''
    score_sim_threshold = 0.82
    
    # Process events in parallel with ThreadPoolExecutor
    def process_event(event):
        besedilo_za_primerjavo = str(event).strip()
        if len(besedilo_za_primerjavo) > 0:
            best_match_content, sim_score = get_most_similar_content(
                df_input, rows_indices, besedilo_za_primerjavo, model, tokenizer)
            return (besedilo_za_primerjavo, best_match_content, sim_score)
        return None
    
    valid_events = [e for e in events if len(str(e).strip()) > 0]
    
    if valid_events:
        # Process in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_event, valid_events))
        
        # Process results
        for result in results:
            if result:
                besedilo, best_match, sim_score = result
                print(Fore.GREEN + besedilo)
                if sim_score <= score_sim_threshold:
                    print("Data quality insufficient. Not reliable input.")
                    return False, ""
                else:
                    best_match_content_all += best_match + '\n'
                    print()
    
    # print(Fore.MAGENTA + "\nGathered input data for the report:\n")
    # print(Fore.MAGENTA + best_match_content_all + "\n")
    
    return True, best_match_content_all
                            
def calc_similarity(df_input, df_traffic_events, window_size):
    tokenizer = AutoTokenizer.from_pretrained("rokn/slovlo-v1", use_fast=False)
    model = AutoModel.from_pretrained("rokn/slovlo-v1")
    
    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Calculate number of rows to process (skipping first 10)
    total_rows = len(df_traffic_events) - 10
    processed_rows = 0
    
    with open('matched_events_quality.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Datum', 'Input_porocilo', 'Porocilo'])
        
        # Create a progress bar (skip first 10 rows)
        pbar = tqdm(total=total_rows, desc="Processing reports", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Initialize batch_results list
        batch_results = []  # Add this line to fix the error
        
        for i, row in df_traffic_events.iterrows():
            if i < 10:
                continue
                
            timestamp = row['Datum']
            # Update progress bar description with current timestamp
            pbar.set_description(f"Processing {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            traffic_dogodki_porocilo = row['Porocilo'].split('Podatki o prometu.')
            best_input_za_porocilo = ""
            
            if len(traffic_dogodki_porocilo) > 1:
                found_data, best_input_za_porocilo = retrieve_surrounding(timestamp, df_input, traffic_dogodki_porocilo[1], window_size, model, tokenizer) 
                if found_data:
                    batch_results.append([str(timestamp), best_input_za_porocilo, row['Porocilo']])
                    
                    # Write in batches of 50
                    if len(batch_results) >= 20:
                        writer.writerows(batch_results)
                        csvfile.flush()
                        batch_results = []

            # Update progress count
            processed_rows += 1
            pbar.update(1)
        
        # Write any remaining batch results before closing
        if batch_results:  # Add this check to write any leftover batches
            writer.writerows(batch_results)
            csvfile.flush()
            
        # Close the progress bar
        pbar.close()
        
        # # Print summary
        # elapsed = datetime.now() - start_time
        # print(f"\nProcessing complete!")
        # print(f"Processed {processed_rows} out of {total_rows} rows in {elapsed}")
        # print(f"Average speed: {processed_rows / max(elapsed.total_seconds(), 1):.2f} rows/second")
            
calc_similarity(df_input_excel, df_output, 50)


