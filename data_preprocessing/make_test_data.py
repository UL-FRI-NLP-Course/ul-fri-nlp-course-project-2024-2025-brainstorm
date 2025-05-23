import pandas as pd
import re
import os

def clean_porocilo_headers(text):
    """
    Remove the first three lines from Porocilo entries:
    - Line with "Prometne informacije" followed by date, time and program info
    - Empty line
    - Line with "Podatki o prometu."
    """
    if not isinstance(text, str):
        return text
    
    # Pattern to match the header lines
    pattern = r'^Prometne informacije.*?\n\n(?:Podatki o prometu\.\n)?'
    
    # Remove the matched pattern
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return cleaned_text.strip()

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
        df['Porocilo'] = df['Porocilo']
        
        # Save the cleaned data
        df.to_csv(output_file, index=False)
        print(f"Data cleaned successfully! Saved to {output_file}")
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

def create_test_dataset():
    """
    Select 20 random rows from the dataset and save them to a test file
    """
    # Get the script directory and construct file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join("/Users/aljazjustin/soal-programi/MAGI/ONJ/ul-fri-nlp-course-project-2024-2025-brainstorm/data_preprocessing/not_dataset.csv")
    output_file = os.path.join("/Users/aljazjustin/soal-programi/MAGI/ONJ/ul-fri-nlp-course-project-2024-2025-brainstorm/data_preprocessing/not_test_data.csv")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Sample 20 random rows
        test_df = df.sample(n=20, random_state=42)  # Using random_state for reproducibility
        
        # Save the test data
        test_df.to_csv(output_file, index=False)
        print(f"Test data created successfully! Saved to {output_file}")
        
        return test_df
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    create_test_dataset()