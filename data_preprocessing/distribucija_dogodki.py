import numpy as np 
import pandas as pd 
import re
import nltk
from nltk.collocations import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_input = pd.read_csv('porocila_input_agg_prompt.csv', encoding='utf-8-sig', sep=';')
df_output = pd.read_csv('porocila_data.csv', encoding='utf-8-sig', sep=';')

df_input['Datum'] = pd.to_datetime(df_input['Datum'])
df_output['Datum'] = pd.to_datetime(df_output['Datum'])

df_input = df_input[df_input['Datum'] > pd.Timestamp(2024,1,1)]
df_output = df_output[df_output['Datum'] > pd.Timestamp(2024,1,1)]

def get_porocilo(i, df_output):
    return df_output.iloc[i]['Porocilo']    

def get_agg_data(i, df_input):
    return df_input.iloc[i]['prompt_podatki']       

traffic_events_dist = []

for idx, row in df_output.iterrows():
    datum = row['Datum']
    porocilo = str(row['Porocilo']).lower()
    podatki_arr = [datum]
    
    for i, keyword in enumerate(['nesreč', 'ovir', 'zaprt', 'žival', 'okvar']):
        podatki_arr.append(porocilo.count(keyword))
    
    traffic_events_dist.append(podatki_arr)       
    
            
    

events_df = pd.DataFrame(traffic_events_dist, columns=['datum', 'nesreč', 'ovir', 'zaprt', 'žival', 'okvar'])
print(events_df.shape)
print(events_df.head(100))

import matplotlib.pyplot as plt 
start_display = 200
nr_display = len(events_df)

plt.plot(events_df['datum'].values[start_display:nr_display], events_df['žival'].values[start_display:nr_display], c='green')
plt.plot(events_df['datum'].values[start_display:nr_display], events_df['okvar'].values[start_display:nr_display], c='black')
plt.plot(events_df['datum'].values[start_display:nr_display], events_df['zaprt'].values[start_display:nr_display], c='blue')
plt.plot(events_df['datum'].values[start_display:nr_display], events_df['ovir'].values[start_display:nr_display], c='gray')
plt.plot(events_df['datum'].values[start_display:nr_display], events_df['nesreč'].values[start_display:nr_display], c='red')
plt.show()

                                      

