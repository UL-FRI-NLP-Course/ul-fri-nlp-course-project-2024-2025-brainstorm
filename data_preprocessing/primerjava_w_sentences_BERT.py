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

from colorama import Fore, Back, Style
from colorama import init
init()

# INPUT AND OUTPUT FILES =============================================================

df_input = pd.read_csv('porocila_input_agg_prompt.csv', encoding='utf-8-sig', sep=';')
df_output = pd.read_csv('porocila_data.csv', encoding='utf-8-sig', sep=';')
df_traffic_events = pd.read_csv('prometni_dogodki_llm.csv', encoding='utf-8-sig', sep=',')

df_input['Datum'] = pd.to_datetime(df_input['Datum'])
df_output['Datum'] = pd.to_datetime(df_output['Datum'])
df_traffic_events['Datum'] = pd.to_datetime(df_traffic_events['Datum'])

start_datum = pd.Timestamp(2022,1,7)
df_input = df_input[df_input['Datum'] > start_datum]
df_output = df_output[df_output['Datum'] > start_datum]
df_traffic_events = df_traffic_events[df_traffic_events['Datum'] > start_datum]

data_dir = 'RTVSlo/RTVSlo'
data_dir = '.'
file_path = os.path.join(data_dir, 'podatki_input.xlsx')
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names[2:]
df_input_excel = pd.concat([excel_file.parse(sheet) for sheet in sheet_names], ignore_index=True)

df_input_excel['Datum'] = pd.to_datetime(df_input_excel['Datum'])
df_input_excel = df_input_excel.sort_values('Datum')

#df_input_excel.drop(columns=[['LegacyId', 'Operater', 'A1', 'B1']], inplace=True, errors='ignore')
df_input_excel = df_input_excel[['Datum', 'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentOpozorilaSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentVremeSLO', 'ContentSplosnoSLO']]
df_input_excel['Datum'] = pd.to_datetime(df_input_excel['Datum'])

# SIMILARITY FUNCTIONS ========================================================

def jaccard_similarity(x,y):
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)   

def tf_idf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return cosine_sim[0][1]

def find_closest_datum_index(df, selected_timestamp):
    time_diffs = (df['Datum'] - selected_timestamp).abs()
    closest_index = time_diffs.idxmin()
    return closest_index

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

def get_most_similar_content(df_input, arr_index, text_A):   
    content_columns = [
        'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentOpozorilaSLO',
        'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentVremeSLO', 
        'ContentSplosnoSLO'
    ]
    
    best_match = {
        'row_index': None,
        'column': None,
        'similarity_score': -1,
        'content_text': None
    }
    
    documents = []
    tokenizer = AutoTokenizer.from_pretrained("rokn/slovlo-v1", use_fast=False)
    model = AutoModel.from_pretrained("rokn/slovlo-v1")
    query = text_A
        
    for idx in arr_index:
        row = df_input.iloc[idx]
        for col in content_columns:
            cell_text = row[col]
            if isinstance(cell_text, str) and cell_text.strip():  
                #vectorizer = TfidfVectorizer()
                #tfidf_matrix = vectorizer.fit_transform([text_A, cell_text])
                #similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                documents.append(cell_text)
                
    # EMBEDDING 
    document_embeddings = [get_embedding(doc, tokenizer, model) for doc in documents]
    query_embedding = get_embedding(query, tokenizer, model)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    
    # FIND THE BEST DOCUMENT 
    nearest_index = np.argmax(similarities)
    best_match = documents[nearest_index]
    print(Fore.BLUE + best_match)
    
    return best_match

def retrieve_surrounding(selected_timestamp, df_input, traffic_dogodki):
    input_row_index = find_closest_datum_index(df_input, selected_timestamp)
   
    events = traffic_dogodki.split('\n')
    if events:
        for event in events:
            besedilo_za_primerjavo = str(event).strip()  
        
            if len(besedilo_za_primerjavo) > 0:
                        
                print(Fore.GREEN + besedilo_za_primerjavo)
                
                i_in_offset = [input_row_index + offset for offset in range(-20, 1)]
                best_match_content = get_most_similar_content(df_input, i_in_offset, besedilo_za_primerjavo)
                
                #print(Fore.MAGENTA + best_match_content)
                print()
                        
def calc_similarity(df_input, df_traffic_events):
    for i, row in df_traffic_events.iterrows():
        timestamp = pd.to_datetime(row['Datum'])
        print(Fore.YELLOW + str(timestamp))
        traffic_dogodki_porocilo = row['Porocilo'].split('Podatki o prometu.')
        
        if len(traffic_dogodki_porocilo)>1:
            retrieve_surrounding(timestamp, df_input, traffic_dogodki_porocilo[1])
    
calc_similarity(df_input_excel, df_output)


