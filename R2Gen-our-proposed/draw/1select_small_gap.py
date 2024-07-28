import pdb

import numpy as np
import pandas as pd
import json
from evaluate import load

from scipy.stats import sem, t, mannwhitneyu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from nltk.tokenize import sent_tokenize


def load_scores_from_json(filename):
    with open(filename, 'r') as f:
        detailed_scores = json.load(f)
    return detailed_scores

def extract_file_names(detailed_scores):
    file_names = list(detailed_scores.keys())
    filtered_file_names = []

    # for file_name in file_names:
    #     match = re.search(r'_\d+$', file_name)
    #     if match:
    #         number = int(match.group(0)[1:])  # 去掉开头的 '_'
    #         # if 'RG' in file_name and number <3:
    #         #     filtered_file_names.append(file_name)
    #         if number <=9:
    #             filtered_file_names.append(file_name)

    return file_names


# 计算并存储每个指标的分差
def compute_scores_and_differences(detailed_scores, key, groups_pair):
    scores_group1 = detailed_scores[key][groups_pair[0]]
    scores_group2 = detailed_scores[key][groups_pair[1]]

    metrics = ['rouge1', 'rouge2', 'rougeL', 'recall', 'precision', 'f1']

    scores_group1_list = [scores_group1[metric] for metric in metrics]
    scores_group2_list = [scores_group2[metric] for metric in metrics]

    average_scores = (np.array(scores_group1_list) + np.array(scores_group2_list)) / 2
    differences = np.abs(np.array(scores_group1_list[:]) - np.array(scores_group2_list[:]))

    return scores_group1_list, scores_group2_list, average_scores, np.mean(differences)



detailed_scores = load_scores_from_json('detailed_scores.json')
file_names = extract_file_names(detailed_scores)

# 遍历目录和编号，计算性能差异
# 存储和排序性能差异
performance_differences = {}
detailed_result_scores = {}
for file_name in file_names:
    scores_difference_groups = []
    scores_average_groups = []
    for group in [['Young', 'Aged'], ['White', 'Black'], ['Female', 'Male']]:
        scores_group1, scores_group2, average_scores, differences = compute_scores_and_differences(detailed_scores, file_name, group)
        scores_difference_groups.append(differences)
        scores_average_groups.append(average_scores)
    average_scores = np.mean(scores_average_groups, axis=0)
    if average_scores[5]>0.28:
        performance_differences[file_name] = np.mean(scores_difference_groups)
        detailed_result_scores[file_name] = (scores_group1, scores_group2, average_scores)

best_combinations = sorted(performance_differences, key=performance_differences.get)[:13]
print(best_combinations)
# for file_name in best_combinations:
#     print(f"File: {file_name}, Average Scores: {detailed_result_scores[file_name][2]}")