import pandas as pd
import os, re
from striprtf.striprtf import rtf_to_text

data_dir = 'RTVSlo/RTVSlo'
tmp_dir =  'RTVSlo/RTVSlo/Podatki - rtvslo.si'
df_porocila = pd.DataFrame(columns=['Datoteka', 'Datum', 'Porocilo'])

def format_datum(match_datum):
    if match_datum is not None:
        dan, mesec, leto, ura, minuta = match_datum.groups()
        if int(dan)==0 or int(mesec)==0 or int(leto)==0:
            print("ERROR: undefined dan, mesec ali leto")
            return f"1. 1. 2000 00.00"
        if int(leto)<100:
            leto = str(int(leto)+2000)
        return f"{dan}. {mesec}. {leto} {ura}.{minuta}"
    else:
        return f"1. 1. 2000 00.00"

for subdir, dirs, files in os.walk(tmp_dir):
    for file in files:
        if file.endswith('.rtf'):
            file_path = os.path.join(subdir, file)
            filename = os.path.basename(file_path).split('.')[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(filename)
                    porocilo_content = rtf_to_text(f.read())
                    match_datum = re.search(r'(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{1,4})\s+(\d{1,2})\.(\d{2})', porocilo_content[0:100])
                    datum_content = format_datum(match_datum)
                    df_porocila = pd.concat([pd.DataFrame([[filename, datum_content, porocilo_content]], columns=df_porocila.columns), df_porocila], ignore_index=True)
            except Exception as e:
                print(e)

df_porocila['Datum'] = pd.to_datetime(df_porocila['Datum'], format='%d. %m. %Y %H.%M')
print(df_porocila.head())
df_porocila = df_porocila.sort_values(by='Datum').reset_index(drop=True)
print(df_porocila.head())

df_porocila.to_csv('porocila_data.csv', index=False, sep=';')

