import pdb

import numpy as np
import pandas as pd
import json
from evaluate import load
from sklearn.metrics import  precision_score, recall_score, f1_score
import re
import os

# 加载并处理数据的函数
def get_group_indices(label_path):
    group_indices = {
        'Male': [],
        'Female': [],
        'White': [],
        'Black': [],
        'Young': [],
        'Aged': []
    }
    with open(label_path, 'r') as file:
        for idx, line in enumerate(file):
            label = json.loads(line)
            if label['sex'] == 'M':
                group_indices['Male'].append(idx)
            if label['sex'] == 'F':
                group_indices['Female'].append(idx)
            if 'WHITE' in label['race'].upper():
                group_indices['White'].append(idx)
            if 'BLACK' in label['race'].upper():
                group_indices['Black'].append(idx)
            if label['age'] <= 65:
                group_indices['Young'].append(idx)
            if label['age'] > 65:
                group_indices['Aged'].append(idx)
    return group_indices

def preprocess_text(text):
    text = re.sub(r'(?i)impression:', '', text)
    text = text.replace('\n', ' ').strip()
    text = " ".join(text.split())
    return text

group_indices = get_group_indices('/home/chenx0c/a100/code/fair/test_label.json')
rouge_metric = load('rouge')

# 计算 ROUGE 分数
def calculate_rouge_scores(directory, group):
    # print(directory+'################')
    with open(directory + 'res.csv', 'r') as f:
        predictions = f.readlines()
        predictions = [preprocess_text(line) for line in predictions]
    with open(directory + 'gts.csv', 'r') as f:
        references = f.readlines()
        references = [preprocess_text(line) for line in references]
    indices = group_indices[group]
    predictions = [predictions[index] for index in indices]
    references = [references[index] for index in indices]

    scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
    return scores['rouge1'], scores['rouge2'], scores['rougeL']

# 计算分类指标
def get_label_accuracy(directory, group):
    df_hyp = pd.read_csv(directory)  # Adjust the actual file path
    df_ref = pd.read_csv(('/home/chenx0c/a100/code/chexpert-labeler/reference.csv'))  # Adjust the actual file path
    indices = group_indices[group]
    df_hyp = df_hyp.iloc[indices]
    df_ref = df_ref.iloc[indices]

    df_hyp_pos1 = (df_hyp == 1).astype(int)
    del df_hyp_pos1["Reports"]
    df_hyp_pos1 = np.array(df_hyp_pos1)
    df_ref_pos1 = (df_ref == 1).astype(int)
    del df_ref_pos1["Reports"]
    df_ref_pos1 = np.array(df_ref_pos1)

    precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
    recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
    f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")
    return recall_pos1, precision_pos1, f1_pos1

def compute_scores_and_differences(decoded_file, chexpert_file, groups):
    detailed_scores = {}
    for group in groups:
        scores_group1 = list(calculate_rouge_scores(decoded_file, group)) + list(get_label_accuracy(chexpert_file, group))
        detailed_scores[group] = {
                'rouge1': scores_group1[0],
                'rouge2': scores_group1[1],
                'rougeL': scores_group1[2],
                'recall': scores_group1[3],
                'precision': scores_group1[4],
                'f1': scores_group1[5]
        }
    return detailed_scores
directories_with_nums = {
    # 'baseline_rank_06_seed4': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # 'baseline_rank_06_seed5': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # 'baseline_rank_06_seed8': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # 'baseline_rank_06_seed9': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    # 'baseline_rank_06_seed1': [2, 3,4,5,6,7,8,9,10,11,12,13,14],
    # 'baseline_rank_06_seed2': [2, 3,4,5,6,7,8,9,10,11,12,13,14],
    # 'baseline_rank_06_seed3': [2, 3,4,5,6,7,8,9,10,11,12,13,14],
    # 'baseline_rank_06_seed7': [2, 3,4,5],
    'baseline_rank_06_seed10': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16],
    'baseline_rank_06_seed11': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16],
    'baseline_rank_06_seed12': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16],
    'baseline_rank_06_seed13': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,15,16],
    # 'baseline_rank_05_seed1': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # 'baseline_rank_05_seed10': [2, 3, 4, 5, 6, 7, 8, 9],
    # 'baseline_rank_05_seed11': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    # 'baseline_rank_05_seed3': [2, 3, 4, 5, 6, 7, 8, 9],
    # 'baseline_rank_05_seed8': [2, 3, 4, 5, 6, 7, 8, 9],
    # 'baseline_rank_08_seed1': [2, 3, 4, 5, 6, 7, 8, 9],
    # 'baseline_rank_08_seed11': [2, 3, 4, 5, 6, 7, 8, 9],

}

all_detailed_scores = {}


for directory, nums in directories_with_nums.items():
    for num in nums:
        print(f'{directory}, {num}')

        decoded_file= f'/home/chenx0c/a100/code/R2Gen-main_brio/results/{directory}/{num}/'
        chexpert_file = f'/home/chenx0c/a100/code/chexpert-labeler/{directory}_{num}.csv'
        groups = ['Young', 'Aged', 'Female', 'Male', 'Black', 'White']
        detailed_scores = compute_scores_and_differences(decoded_file, chexpert_file, groups)
        all_detailed_scores[f'{directory}_{num}'] = detailed_scores

def save_scores_to_json(detailed_scores, filename):
    with open(filename, 'w') as f:
        json.dump(detailed_scores, f, indent=4)
save_scores_to_json(all_detailed_scores, 'detailed_scores2.json')
