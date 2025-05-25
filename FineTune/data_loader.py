import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import re
import pandas as pd  # ADD THIS LINE

class TrafficReportDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Clean HTML tags if present
        input_text = re.sub(r'<.*?>', '', item['Input_porocilo'])
        input_text = re.sub(r'\s+', ' ', input_text).strip()
        
        # Handle empty inputs
        if len(input_text.strip()) < 10:
            input_text = "Ni posebnih prometnih podatkov."
        
        # Create the user message
        user_message = f"""Si strokovnjak za prometna poročila za slovenski radio. Ustvari kratko prometno poročilo iz podanih podatkov.

    PRAVILA:
    - Pri več dogodkih: najprej najpomembnejše (zapore, nesreče)
    - Uporabljaj kratke stavke, primerne za radio
    - Maksimalno 60 sekund branja
    - Uporabljaj uradni prometni jezik

    STRUKTURA (prilagodi glede na podatke):
    - Glavne zapore/motnje
    - Obvozi
    - Omejitve za tovorna vozila  
    - Vremenske razmere (če vplivajo)

    Podatki o prometu: {input_text}"""

        if 'Porocilo' in item and item['Porocilo']:
            # Format for training with proper GaMS conversation format
            full_conversation = f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n{item['Porocilo']}<end_of_turn><eos>"
            
            encoding = self.tokenizer(full_conversation,
                                    max_length=self.max_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
            
            input_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]
            labels = input_ids.clone()
            
            # Find where model response starts
            user_part = f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
            user_encoding = self.tokenizer(user_part, add_special_tokens=False, return_tensors="pt")
            input_length = len(user_encoding.input_ids[0])
            
            # Mask user input in labels (only train on model output)
            labels[:input_length] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        
        # For inference
        else:
            inference_prompt = f"<bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
            
            encoding = self.tokenizer(inference_prompt,
                                    max_length=self.max_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")
            
            return {
                "input_ids": encoding.input_ids[0],
                "attention_mask": encoding.attention_mask[0]
            }

def load_and_prepare_data(csv_path, tokenizer, test_size=0.05, seed=42):
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
    train_data, val_data = train_test_split(processed_data, test_size=0.05, random_state=seed)
    
    return train_data, val_data

def create_dataloaders(train_data, val_data, tokenizer, batch_size, max_length=512):
    """
    Create PyTorch DataLoaders for training and validation
    """
    train_dataset = TrafficReportDataset(train_data, tokenizer, max_length)
    val_dataset = TrafficReportDataset(val_data, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=8 ,pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,num_workers=4)
    
    return train_dataloader, val_dataloader