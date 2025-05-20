import numpy as np 
import pandas as pd 
import re, os

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

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    return mean_pooled.squeeze().numpy()

# ================================================================================

def preprocess_html(text):
    arr = re.split(r'</?p>', text)
    return [el.strip() for el in arr if el.strip()]

def get_most_similar_content(df_input, rows_indices, query, model, tokenizer):   
    content_columns = ['A1', 'B1',
        'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentOpozorilaSLO',
        'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentVremeSLO', 
        'ContentSplosnoSLO']
        
    documents = []
        
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
                
    # EMBEDDING 
    document_embeddings = [get_embedding(doc, tokenizer, model) for doc in documents]
    query_embedding = get_embedding(query, tokenizer, model)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]

    # FIND THE BEST DOCUMENT 
    top_k = 3
    top_indices = similarities.argsort()[-top_k:][::-1]
    best_match = documents[top_indices[0]]
    top_sim = similarities[top_indices[0]]
    print(Fore.BLUE + f"Score: {top_sim:.4f}:\nMatch: {best_match}\n")
    return best_match, top_sim

                
def retrieve_surrounding(selected_timestamp, df_input, traffic_dogodki, window_size, model, tokenizer):
    df_prior = df_input[df_input['Datum'] < selected_timestamp]
    rows_within_window = df_prior.tail(window_size)
    #rows_within_window = rows_within_window[rows_within_window['Datum'] >= (selected_timestamp - timedelta(hours=1.5))]
    
    if rows_within_window.empty:
        print(Fore.RED + "No or incorrect input data.")
        return

    rows_indices = rows_within_window.index.tolist()
    events = traffic_dogodki.split('\n')
    best_match_content_all = ''
    score_sim_threshold = 0.82
      
    if events:
        for event in events:
            besedilo_za_primerjavo = str(event).strip()
            if len(besedilo_za_primerjavo) > 0:
                print(Fore.GREEN + besedilo_za_primerjavo)
                best_match_content, sim_score = get_most_similar_content(df_input, rows_indices, besedilo_za_primerjavo, model, tokenizer)
                if sim_score <= score_sim_threshold:
                    print("Data quality insufficient. Not reliable input.")
                    return False, ""
                else:
                    best_match_content_all += best_match_content + '\n'
                    print()
    #best_match_content_all += '"'
    print(Fore.MAGENTA + "\nGathered input data for the report:\n")
    print(Fore.MAGENTA +  best_match_content_all + "\n")
    
    return True, best_match_content_all
                            
def calc_similarity(df_input, df_traffic_events, window_size):
    tokenizer = AutoTokenizer.from_pretrained("rokn/slovlo-v1", use_fast=False)
    model = AutoModel.from_pretrained("rokn/slovlo-v1")
    
    with open('matched_events_quality.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Datum', 'Input_porocilo', 'Porocilo'])
        
        for i, row in df_traffic_events.iterrows():
            if i<10:
                continue
            timestamp = row['Datum']
            print(Fore.YELLOW + str(timestamp))
            traffic_dogodki_porocilo = row['Porocilo'].split('Podatki o prometu.')
            best_input_za_porocilo = ""
            if len(traffic_dogodki_porocilo)>1:
                found_data, best_input_za_porocilo = retrieve_surrounding(timestamp, df_input, traffic_dogodki_porocilo[1], window_size, model, tokenizer) 
                if found_data:
                    writer.writerow([str(timestamp), best_input_za_porocilo, row['Porocilo']])

            current_time = datetime.now()
            if current_time.hour == 8 and current_time.minute == 20:
                print("It's 8:20 - stopping execution.")
                break
            
            if keyboard.is_pressed('t'):
                print("Stopping.")
                break
            
calc_similarity(df_input_excel, df_output, 50)


