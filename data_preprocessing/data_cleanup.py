import pandas as pd
import re
import os

def clean_porocilo_headers(text):
    """
    Remove the first four lines from Porocilo entries
    """
    if not isinstance(text, str):
        return text
    
    # Split text into lines
    lines = text.split('\n')
    
    # Remove first 4 lines if there are at least 4 lines
    if len(lines) > 4:
        cleaned_lines = lines[4:]
        return '\n'.join(cleaned_lines).strip()
    else:
        # If less than 4 lines, return empty string or original text
        return text.strip()

def clean_dataset():
    """
    Read the CSV file, clean the Porocilo column, and save to a new file
    """
    # Get the script directory and construct file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    input_file = os.path.join("/Users/aljazjustin/soal-programi/MAGI/ONJ/ul-fri-nlp-course-project-2024-2025-brainstorm/data_preprocessing/not_dataset.csv")
    output_file = os.path.join("/Users/aljazjustin/soal-programi/MAGI/ONJ/ul-fri-nlp-course-project-2024-2025-brainstorm/data_preprocessing/cleaned_dataset.csv")
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Clean the Porocilo column
        df['Porocilo'] = df['Porocilo'].apply(clean_porocilo_headers)
        
        # Save the cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully! Saved to {output_file}")
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    clean_dataset()