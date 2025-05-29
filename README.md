# Natural language processing course: Automatic Generation of Slovenian Traffic News for RTV Slovenija

This project develops an automated system for generating Slovenian-language radio traffic reports for RTV Slovenija using Large Language Models (LLMs). The system transforms raw traffic event data into stylistically consistent and factually accurate reports suitable for radio broadcast.

## Project Overview

The goal is to automate the manual process currently used by RTV Slovenija, where students check and type traffic reports every 30 minutes. The system uses fine-tuned LLMs with prompt engineering to generate short, radio-appropriate traffic reports from Excel data provided by the promet.si portal.

## Key Features

- **Multi-stage Data Preprocessing**: Sophisticated pipeline to align raw traffic data with compiled RTV reports
- **Fine-tuned GaMS-9B Model**: Slovenian-specific model optimized for traffic report generation using LoRA (Low-Rank Adaptation)
- **HPC Cluster Integration**: Distributed training and inference on high-performance computing infrastructure
- **Comprehensive Evaluation**: Both quantitative metrics (BLEU, ROUGE, METEOR) and LLM-based heuristic assessments
- **Rule-based Post-processing**: Automated corrections for grammar, terminology, and style consistency

## Project Structure

```
├── data_preprocessing/          # Data cleaning and preparation scripts
│   ├── data_cleanup.py         # Main data cleaning utilities
│   ├── make_test_data.py       # Test dataset creation
│   ├── primerjava_w_sentences_BERT_w_QA.py  # BERT-based semantic matching
│   └── LLM_porocilo_extraction.py  # Event extraction using Gemini
├── FineTune/                   # Fine-tuning implementation
│   ├── finetune_GaMS_9B.py     # Main fine-tuning script
│   ├── params_GaMS_9B.py       # Configuration and hyperparameters
│   ├── data_loader.py          # Data loading utilities
│   ├── fine_tune.sh            # SLURM batch script for training
│   └── Instructions.ipynb      # Detailed setup instructions
├── Generate_Report/            # Report generation and testing
│   ├── report_generation.py    # Main generation script
│   └── run_rep_gen.sh          # SLURM batch script for testing
├── final_report_generation/    # Final optimized generation pipeline
├── gemini-api/                 # Gemini API integration for prompting
├── HPC/                        # HPC setup and configuration
│   ├── HowToHPC.ipynb          # HPC usage instructions
│   └── LLama_17B/              # Large model experiments
├── evaluation_of_generated/    # Evaluation scripts and metrics
├── report/                     # Academic report (LaTeX)
├── previous_tesing/            # Archive of early experiments and tests
└── Report_Generation/          # Additional generation utilities
```

## Methodology

### 1. Data Preprocessing
- **Event Extraction**: Used Gemini 2.0 Flash to extract structured events from RTV reports
- **Semantic Matching**: Employed Slovene BERT model (`rokn/slovlo-v1`) to match raw traffic data with compiled reports
- **Data Alignment**: Created input-output pairs for training by aggregating relevant traffic events

### 2. Model Selection and Fine-tuning
- **Base Model**: GaMS-9B-Instruct (Slovenian-specific LLM)
- **Fine-tuning Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA
- **Training Environment**: HPC cluster with GPU acceleration
- **Optimization**: Custom training loop with gradient accumulation and learning rate scheduling

### 3. Prompt Engineering
- **Initial Exploration**: Tested various models (Gemini, Mistral-AI, Llama, DeepSeek)
- **Prompt Evolution**: From detailed API prompts to concise fine-tuning prompts
- **Style Guidelines**: Incorporated RTV Slovenija's formatting and style requirements

### 4. Evaluation Framework
- **Quantitative Metrics**: BLEU, ROUGE-1/2/L, METEOR scores
- **Qualitative Assessment**: LLM-based heuristic evaluation using Gemini 2.0 Flash
- **Parameter Optimization**: Tested multiple generation configurations (temperature, repetition penalty)

## Key Results

The fine-tuned GaMS-9B model achieved optimal performance with:
- **Temperature**: 0.4
- **Repetition Penalty**: 1.2
- **ROUGE-L F1 Score**: 0.593
- **High qualitative ratings** for factual accuracy and stylistic appropriateness

## Usage

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- Access to HPC cluster (for training)

### Quick Start
1. **Data Preparation**:
   ```bash
   cd data_preprocessing
   python data_cleanup.py
   python make_test_data.py
   ```

2. **Fine-tuning** (on HPC):
   ```bash
   cd FineTune
   sbatch fine_tune.sh
   ```

3. **Generate Reports**:
   ```bash
   cd Generate_Report
   sbatch run_rep_gen.sh
   ```

### Configuration
Model and training parameters can be adjusted in [`FineTune/params_GaMS_9B.py`](FineTune/params_GaMS_9B.py).

## Evaluation

The system is evaluated using:
- **Lexical Similarity**: BLEU, ROUGE, METEOR metrics against reference reports
- **Content Quality**: Factual correctness, location accuracy, event type identification
- **Style Assessment**: Conciseness, grammatical correctness, radio suitability

## Future Work

- **Model Refinement**: Explore advanced fine-tuning techniques and parameter optimization
- **Content Enhancement**: Improve relevance filtering and domain-specific understanding
- **Real-time Integration**: Develop pipeline for production deployment at RTV Slovenija
- **Multilingual Extension**: Add translation capabilities for broader accessibility

## Data Sources

- **Input Data**: Traffic event data from promet.si portal (Excel format)
- **Target Output**: RTV Slovenija radio traffic reports
- **Guidelines**: Official formatting and style instructions for human reporters

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Academic Report

A comprehensive academic report detailing the methodology, evaluation, and results is available in the [`report/`](report/) directory.

## Notes

- All early experiments and prototype code are archived in [`previous_tesing/`](previous_tesing/)
- HPC-specific setup instructions are available in [`HPC/HowToHPC.ipynb`](HPC/HowToHPC.ipynb)
- Detailed fine-tuning instructions can be found in [`FineTune/Instructions.ipynb`](FineTune/Instructions.ipynb)

---

**Authors**: Aljaž Justin, Edin Ćehić, Lea Briški  
**Advisor**: Slavko Žitnik  
**Institution**: University of Ljubljana, Faculty of Computer and Information Science  
**Course**: Natural Language Processing (2024/2025)