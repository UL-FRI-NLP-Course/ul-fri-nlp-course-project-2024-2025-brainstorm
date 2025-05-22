import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import re

class TrafficReportDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Use the Input_porocilo as input
        # Clean HTML tags if present
        input_text = re.sub(r'<.*?>', '', item['Input_porocilo'])
        
        
        shortened_prompt = """Generiraj prometno poročilo v slovenščini za radio.
            Upoštevaj standardno strukturo:
            1. Začni z "Prometne informacije [datum] [čas] za Radio Slovenija"
            2. Vključi samo pomembne prometne dogodke
            3. Uporabljaj trpne oblike in ustrezno terminologijo
            4. Poročilo naj bo jedrnato (<1 min branja)

            Podatki o prometu:
            """        
        # Format input with appropriate prompt
        input_prompt = f"{shortened_prompt}\n{input_text}"
        
        # For training we include the target report (Porocilo)
        if 'Porocilo' in item and item['Porocilo']:
            # Format full input-output sequence
            input_output = f"{input_prompt}\n\nReport:\n{item['Porocilo']}"
            
            encoding = self.tokenizer(input_output, 
                                     max_length=self.max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            
            # Create input_ids and labels for training
            input_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]
            
            # For causal language modeling, labels are the same as input_ids
            # but we set the input part to -100 to ignore in loss calculation
            labels = input_ids.clone()
            
            # Determine the position where the output starts
            input_encoding = self.tokenizer(input_prompt, return_tensors="pt")
            input_length = len(input_encoding.input_ids[0])
            
            # Set labels for input portion to -100 (ignore in loss calculation)
            labels[:input_length] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # For inference, we only need input
        else:
            encoding = self.tokenizer(input_prompt, 
                                     max_length=self.max_length,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            
            return {
                "input_ids": encoding.input_ids[0],
                "attention_mask": encoding.attention_mask[0]
            }

def load_and_prepare_data(csv_path, tokenizer, test_size=0.1, seed=42):
    """
    Load data from CSV and prepare for training/testing
    """
    # Load CSV data with the correct columns
    df = pd.read_csv(csv_path)
    
    # Process data into the expected format
    processed_data = []
    
    for _, row in df.iterrows():
        # Clean and process HTML tags in the input and report
        input_text = re.sub(r'<.*?>', '', str(row['Input_porocilo']))
        report_text = str(row['Porocilo'])
        
        processed_data.append({
            'Datum': row['Datum'],
            'Input_porocilo': input_text,
            'Porocilo': report_text
        })
    
    # Split data for training and validation
    train_data, val_data = train_test_split(processed_data, test_size=test_size, random_state=seed)
    
    return train_data, val_data

def create_dataloaders(train_data, val_data, tokenizer, batch_size, max_length=512):
    """
    Create PyTorch DataLoaders for training and validation
    """
    train_dataset = TrafficReportDataset(train_data, tokenizer, max_length)
    val_dataset = TrafficReportDataset(val_data, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_dataloader, val_dataloader