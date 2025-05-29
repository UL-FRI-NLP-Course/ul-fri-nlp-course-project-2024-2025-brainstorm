"""
Configuration parameters for fine-tuning GaMS-9B model on traffic reports.
"""

import os

#######################
# MODEL CONFIGURATION #
#######################

# Base model paths
MODEL_PATH = "/d/hpc/projects/onj_fri/brainstorm/models/GaMS-9B-Instruct"
MODEL_NAME = "GaMS-9B-Instruct"
BASE_MODEL = "google/gemma-2-9b"

# Model settings
MODEL_PRECISION = "float16"  # Options: "float32", "float16", "bfloat16"
GRADIENT_CHECKPOINTING = True  # Saves memory but slows down training

#########################
# TRAINING HYPERPARAMETERS #
#########################

# Basic training parameters
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 512
WARMUP_RATIO = 0.1  # Percentage of steps for warmup
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 8  # Increase if you need larger effective batch size

# Optimizer and scheduler
OPTIMIZER = "adamw"  # Options: "adamw", "adafactor"
LR_SCHEDULER = "linear"  # Options: "linear", "cosine", "constant", "constant_with_warmup"

#######################
# LORA CONFIGURATION #
#######################

LORA_CONFIG = {
    "r": 16,                    # Rank of the update matrices
    "lora_alpha": 32,           # Scaling factor for trained weights
    "lora_dropout": 0.1,       # Dropout probability for LoRA layers
    "bias": "none",             # Whether to train bias parameters 
    "task_type": "CAUSAL_LM",   # Task type
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

#########################
# DATASET CONFIGURATION #
#########################

# Dataset paths and settings
DARASET_DIR = "/d/hpc/projects/onj_fri/brainstorm/not_dataset_folder/cleaned_dataset.csv"
TEST_SPLIT = 0.05
RANDOM_SEED = 42

# Text processing
MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 128

#################################
# OUTPUT/LOGGING CONFIGURATION #
#################################

# Output directories
BASE_OUTPUT_DIR = "/d/hpc/projects/onj_fri/brainstorm/FineTune_models"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/GaMS-9B-Instruct-1"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
LOGGING_DIR = "/d/hpc/home/aj3477/NLP/logs/GaMS-9B-Instruct"
TESTING_DIR = f"{LOGGING_DIR}/testing"

# Evaluation settings
EVAL_STEPS = 500  # Run evaluation every N steps
SAVE_STEPS = 800  # Save checkpoint every N steps

# Wandb configuration
WANDB_PROJECT = "traffic-report-generation"
WANDB_ENABLED = True

# Generation parameters
GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "num_beams": 1,
    "do_sample": True,  # Make sure this is True when using temperature and top_p
    "max_new_tokens": 256,
    "repetition_penalty": 1.2
}