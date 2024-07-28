import pdb
from scipy import stats
import pandas as pd
import numpy as np
from scipy.stats import sem, t

from evaluate import load

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os
rouge_metric = load('rouge')


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


group_indices = get_group_indices('/home/chenx0c/a100/code/fair/test_label.json')


def calculate_rouge_scores(decoded_template, group):
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    for directory in decoded_template:
        f=open(directory)
        predictions = f.readlines()
        f=open('/home/chenx0c/a100/code/R2Gen-main/results/RGen8/3/gts.csv')
        references = f.readlines()
        indices = group_indices[group]
        predictions = [predictions[index] for index in indices]
        references = [references[index] for index in indices]
        scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
        rouge1 = scores['rouge1']
        rouge2 = scores['rouge2']
        rougel = scores['rougeL']
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
        rougel_scores.append(rougel)
    return rouge1_scores, rouge2_scores, rougel_scores


def get_label_accuracy(hypothesis, group):
    f1s = []
    recalls = []
    precisions = []

    for directory in hypothesis:
        df_hyp = pd.read_csv(directory)
        df_ref = pd.read_csv('/home/chenx0c/a100/code/chexpert-labeler/reference.csv')
        indices = group_indices[group]
        df_hyp = df_hyp.iloc[indices]
        df_ref = df_ref.iloc[indices]

        df_hyp_pos1 = (df_hyp == 1).astype(int)
        del df_hyp_pos1["Reports"]
        df_hyp_pos1 = np.array(df_hyp_pos1)
        df_ref_pos1 = (df_ref == 1).astype(int)
        del df_ref_pos1["Reports"]
        df_ref_pos1 = np.array(df_ref_pos1)

        df_hyp_0 = (df_hyp == 0).astype(int)
        del df_hyp_0["Reports"]
        df_hyp_0 = np.array(df_hyp_0)
        df_ref_0 = (df_ref == 0).astype(int)
        del df_ref_0["Reports"]
        df_ref_0 = np.array(df_ref_0)

        df_hyp_neg1 = (df_hyp == -1).astype(int)
        del df_hyp_neg1["Reports"]
        df_hyp_neg1 = np.array(df_hyp_neg1)
        df_ref_neg1 = (df_ref == -1).astype(int)
        del df_ref_neg1["Reports"]
        df_ref_neg1 = np.array(df_ref_neg1)

        df_hyp_all = df_hyp_pos1 + df_hyp_0 + df_hyp_neg1
        df_ref_all = df_ref_pos1 + df_ref_0 + df_ref_neg1

        # Precision
        precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
        precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro")
        precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro")
        precision_all = precision_score(df_ref_all, df_hyp_all, average="micro")

        # Recall
        recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
        recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro")
        recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro")
        recall_all = recall_score(df_ref_all, df_hyp_all, average="micro")

        # F1
        f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro", zero_division=1)
        f1_0 = f1_score(df_ref_0, df_hyp_0, average="macro", zero_division=1)
        f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro", zero_division=1)
        f1_all = f1_score(df_ref_all, df_hyp_all, average="micro", zero_division=1)  # 0.0014454147043923005

        # f1s.append(f1_neg1)
        # recalls.append(recall_neg1)
        # precisions.append(precision_neg1)

        # f1s.append(f1_all)
        # recalls.append(recall_all)
        # precisions.append(precision_all)

        f1s.append(f1_pos1)
        recalls.append(recall_pos1)
        precisions.append(precision_pos1)
    return recalls, precisions, f1s


groups = ['Female', 'Male', 'Black', 'White', 'Young', 'Aged']
results = {group: {} for group in groups}


def conf_interval(data):
    n = len(data)
    stderr = sem(data)
    h = stderr * t.ppf((1 + 0.95) / 2, n - 1)
    return h

#none
baseline_directories_with_nums = {
    'RGen8': [ 4],
    'RGen3': [7],
    'RGen11': [2],
    'RGen2': [5, 4],
    'RGen4': [5, 3],
}


directories_with_nums = {
    'baseline_rank_06_seed8': [14],
    'baseline_rank_06_seed9': [14],
    'baseline_rank_06_seed1': [7],
    'baseline_rank_06_seed3': [13, 7],
    'baseline_rank_06_seed12': [16],
    'baseline_rank_06_seed13': [6],
}



decoded_file_template_1 = [
    f'/home/chenx0c/a100/code/R2Gen-main/results/{directory}/{num}/res.csv'
    for directory, nums in baseline_directories_with_nums.items()
    for num in nums
]

decoded_file_template_2 = [
    f'/home/chenx0c/a100/code/R2Gen-main_brio/results/{directory}/{num}/res.csv'
    for directory, nums in directories_with_nums.items()
    for num in nums
]

chexpert_path1 = [
    f'/home/chenx0c/a100/code/chexpert-labeler/{directory}_{num}.csv'
    for directory, nums in baseline_directories_with_nums.items()
    for num in nums
]

chexpert_path2 = [
    f'/home/chenx0c/a100/code/chexpert-labeler/{directory}_{num}.csv'
    for directory, nums in directories_with_nums.items()
    for num in nums
]

for group in groups:
    results[group]['rouge_scores1_base'], results[group]['rouge_scores2_base'], results[group][
        'rouge_scoresl_base'] = calculate_rouge_scores(decoded_file_template_1, group)
    results[group]['rouge_scores1_our'], results[group]['rouge_scores2_our'], results[group][
        'rouge_scoresl_our'], = calculate_rouge_scores(decoded_file_template_2, group)
    results[group]['recall_scores_base'], results[group]['precision_scores_base'], results[group][
        'f1_scores_base'] = get_label_accuracy(chexpert_path1, group)
    results[group]['recall_scores_our'], results[group]['precision_scores_our'], results[group][
        'f1_scores_our'] = get_label_accuracy(chexpert_path2, group)

metrics = ['rouge_scores1_base', 'rouge_scores1_our',
           'rouge_scores2_base', 'rouge_scores2_our',
           'rouge_scoresl_base', 'rouge_scoresl_our',
           'recall_scores_base', 'recall_scores_our',
           'precision_scores_base', 'precision_scores_our',
           'f1_scores_base', 'f1_scores_our']
differences = []
conf_intervals = []
all_scores = []

def compute_differences_and_plot(results, check1, check2):
    filename=f'blue_{check1}_vs_{check2}_1.pdf'
    metrics = ['rouge_scores1_base', 'rouge_scores1_our',
               'rouge_scores2_base', 'rouge_scores2_our',
               'rouge_scoresl_base', 'rouge_scoresl_our',
               'recall_scores_base', 'recall_scores_our',
               'precision_scores_base', 'precision_scores_our',
               'f1_scores_base', 'f1_scores_our']#1 2  代表baseline 和our
    all_diff_scores = []
    all_scores = []
    for metric in metrics:
        all_scores.append(np.array(results[check1][metric]))

    for metric in metrics:
        group1_scores = np.array(results[check1][metric])
        group2_scores = np.array(results[check2][metric])
        diff = np.abs(group1_scores - group2_scores)
        all_diff_scores.append(diff)


    # Plot setup
    age_categories = ['Baseline', 'Proposed']
    colors = ['#74a9cf', '#0570b0']
    plt.figure(figsize=(6, 3.8))
    bar_width = 0.35
    space_between_groups = 0.3

    for i in range(6):  # Assuming the first 6 metrics are the ones we plot
        for j, cat in enumerate(age_categories):
            data = all_diff_scores[i * 2 + j]
            mean = np.mean(data)
            semm = stats.sem(data)
            ci = stats.t.interval(0.8, len(data) - 1, loc=mean, scale=semm)
            bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
            plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=5)

        stat, p_value = stats.mannwhitneyu(all_diff_scores[i * 2], all_diff_scores[i * 2 + 1],alternative='greater')
        if p_value < 0.05:  # Check if the results are statistically significant
            max_mean = max(np.mean(all_diff_scores[i * 2]), np.mean(all_diff_scores[i * 2 + 1]))
            plt.text(i * (len(age_categories) * bar_width + space_between_groups)+ bar_width/2, max_mean + 0.01, '*',
                     fontsize=15, ha='center')
            print(f'{check1} {i} yes')
        else:
            print(f'{check1} {i} no')


    # X-axis and labels
    xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(6)]
    xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
    plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=10)
    plt.ylabel('MFD', fontsize=15)
    plt.ylim(0, 0.08)
    plt.title(f'{check1.capitalize()} vs {check2.capitalize()}')
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors], [cat.capitalize() for cat in age_categories], fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)

    # 绘制柱状图
    plt.figure(figsize=(6, 3.8))
    bar_width = 0.3
    colors=['#b9d9b7','#7da97a']

    for i in range(6):  # Assuming the first 6 metrics are the ones we plot
        for j, cat in enumerate(age_categories):
            data = all_scores[i * 2 + j]
            mean = np.mean(data)
            semm = stats.sem(data)
            ci = stats.t.interval(0.8, len(data) - 1, loc=mean, scale=semm)
            bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
            plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]], capsize=5)
        stat, p_value = stats.mannwhitneyu(all_scores[i * 2], all_scores[i * 2 + 1],alternative='greater')
        if p_value < 0.05:  # Check if the results are statistically significant
            max_mean = max(np.mean(all_scores[i * 2]), np.mean(all_scores[i * 2 + 1]))
            plt.text(i * (len(age_categories) * bar_width + space_between_groups)+ bar_width/2, max_mean + 0.01, '*',
                     fontsize=15, ha='center')
            print(f'{check1} {i} yes')
        else:
            print(f'{check1} {i} no')

    xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (
                len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(6)]
    xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
    plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=10)
    plt.ylabel('Score', fontsize=15)
    plt.title(f'{check1.capitalize()}')
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               [cat.capitalize() for cat in age_categories], fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'green_{check1}.pdf')

group_pairs = [('Young', 'Aged'), ('Female', 'Male'), ('White', 'Black')]
for i, (check1, check2) in enumerate(group_pairs):
    compute_differences_and_plot(results, check1, check2)

