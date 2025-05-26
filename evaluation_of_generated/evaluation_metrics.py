import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import pandas as pd


FILE_NAME = "generated_reports_57976558.json"
EXTRACTED_FILE_NUMBER = FILE_NAME.split("_")[-1].split(".")[0]

print(f"Evaluating reports from file: {FILE_NAME} with extracted file number: {EXTRACTED_FILE_NUMBER}")

df_results = pd.read_json(FILE_NAME)
col_names = ['timestamp', 'events', 'generated_report', 'target_report']

def calc_bleu_score(true_report, generated_report):
    reference = true_report.split()
    candidate = generated_report.split()

    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return bleu_score

def calc_rogue_score(true_report, generated_report):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(true_report, generated_report)
    return scores

def calc_meteor_score(true_report, generated_report):
    reference = word_tokenize(true_report)
    candidate = word_tokenize(generated_report)
    return meteor_score([reference], candidate)

def evaluate_reports(df_results):
    df_results['bleu_score'] = df_results.apply(lambda row: calc_bleu_score(row['target_report'], row['generated_report']), axis=1)
    df_results['rouge_score'] = df_results.apply(lambda row: calc_rogue_score(row['target_report'], row['generated_report']), axis=1)
    df_results['meteor_score'] = df_results.apply(lambda row: calc_meteor_score(row['target_report'], row['generated_report']), axis=1)

    return df_results[['timestamp', 'events', 'generated_report', 'target_report', 'bleu_score', 'rouge_score', 'meteor_score']]
    
def evaluate_reports_scores_only(df_results):
    df_results['bleu_score'] = df_results.apply(lambda row: calc_bleu_score(row['target_report'], row['generated_report']), axis=1)
    df_results['rouge_score'] = df_results.apply(lambda row: calc_rogue_score(row['target_report'], row['generated_report']), axis=1)
    df_results['meteor_score'] = df_results.apply(lambda row: calc_meteor_score(row['target_report'], row['generated_report']), axis=1)

    return df_results[['timestamp', 'bleu_score', 'rouge_score', 'meteor_score']]

def calc_metrics_avg(df_results):
    avg_bleu = df_results['bleu_score'].mean()
    avg_meteor = df_results['meteor_score'].mean()
    
    rouge1_f = df_results['rouge_score'].apply(lambda x: x['rouge1'].fmeasure).mean()
    rouge2_f = df_results['rouge_score'].apply(lambda x: x['rouge2'].fmeasure).mean()
    rougel_f = df_results['rouge_score'].apply(lambda x: x['rougeL'].fmeasure).mean()

    avg_rouge = {
        'rouge1_f': rouge1_f,
        'rouge2_f': rouge2_f,
        'rougeL_f': rougel_f
    }

    return avg_bleu, avg_rouge, avg_meteor
    
df_results = evaluate_reports(df_results)
avg_bleu, avg_rouge, avg_meteor = calc_metrics_avg(df_results)
print(f"Average BLEU Score: {avg_bleu}")    
print(f"Average ROUGE Score: ROUGE1: {str(avg_rouge['rouge1_f'])}, ROUGE2: {str(avg_rouge['rouge2_f'])}, ROUGEL: {str(avg_rouge['rougeL_f'])}")
print(f"Average METEOR Score: {avg_meteor}")

# SAVE 
df_results.to_json(f'evaluation_results-{EXTRACTED_FILE_NUMBER}.json', orient="records", lines=True)



