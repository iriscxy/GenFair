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
import json


# 读取 JSON 文件
def load_scores_from_json(filename):
    with open(filename, 'r') as f:
        detailed_scores = json.load(f)
    return detailed_scores


# 从给定的分数数据和文件名列表中提取数据
def load_data_from_json(base_scores, our_scores, base_file_names, our_file_names, groups):
    data_dict = {}

    # 将 base_scores 和 our_scores 组织到 data_dict 中
    for base_name in base_file_names:
        data_dict[base_name] = {
            'base': base_scores.get(base_name, {}),
            'our': {}
        }

    for our_name in our_file_names:
        if our_name in data_dict:
            data_dict[our_name]['our'] = our_scores.get(our_name, {})
        else:
            data_dict[our_name] = {
                'base': {},
                'our': our_scores.get(our_name, {})
            }

    results = {group: {} for group in groups}

    # 初始化 results 字典
    for group in groups:
        results[group]['rouge_scores1_base'] = []
        results[group]['rouge_scores2_base'] = []
        results[group]['rouge_scoresl_base'] = []
        results[group]['rouge_scores1_our'] = []
        results[group]['rouge_scores2_our'] = []
        results[group]['rouge_scoresl_our'] = []
        results[group]['recall_scores_base'] = []
        results[group]['precision_scores_base'] = []
        results[group]['f1_scores_base'] = []
        results[group]['recall_scores_our'] = []
        results[group]['precision_scores_our'] = []
        results[group]['f1_scores_our'] = []

        # 遍历 data_dict 中的每个文件
        for file_name, scores in data_dict.items():
            if group in scores['base']:
                base_scores_group = scores['base'][group]
                results[group]['rouge_scores1_base'].append(base_scores_group.get('rouge1', 0))
                results[group]['rouge_scores2_base'].append(base_scores_group.get('rouge2', 0))
                results[group]['rouge_scoresl_base'].append(base_scores_group.get('rougeL', 0))
                results[group]['recall_scores_base'].append(base_scores_group.get('recall', 0))
                results[group]['precision_scores_base'].append(base_scores_group.get('precision', 0))
                results[group]['f1_scores_base'].append(base_scores_group.get('f1', 0))

            if group in scores['our']:
                our_scores_group = scores['our'][group]
                results[group]['rouge_scores1_our'].append(our_scores_group.get('rouge1', 0))
                results[group]['rouge_scores2_our'].append(our_scores_group.get('rouge2', 0))
                results[group]['rouge_scoresl_our'].append(our_scores_group.get('rougeL', 0))
                results[group]['recall_scores_our'].append(our_scores_group.get('recall', 0))
                results[group]['precision_scores_our'].append(our_scores_group.get('precision', 0))
                results[group]['f1_scores_our'].append(our_scores_group.get('f1', 0))

    return results


# 示例调用
base_scores = load_scores_from_json('/home/chenx0c/a100/code/R2Gen-main/draw/detailed_scores.json')
our_scores = load_scores_from_json('detailed_scores.json')
base_file_names = ['RGen3_7', 'RGen8_5', 'RGen15_6', 'RGen10_10', 'RGen10_12', 'RGen9_5', 'RGen4_6', 'RGen10_9',
                   'RGen17_7', 'RGen13_13', 'RGen10_7']
our_file_names = ['baseline_rank_08_seed8_7', 'baseline_rank_08_seed8_9', 'baseline_rank_06_seed8_14',
                  'baseline_rank_06_seed9_14', 'baseline_rank_06_seed13_5', 'baseline_rank_06_seed10_13',
                  'baseline_rank_06_seed3_5', 'baseline_rank_06_seed3_13', 'baseline_rank_06_seed11_15']
groups = ['Female', 'Male', 'Black', 'White', 'Young', 'Aged']

results = load_data_from_json(base_scores, our_scores, base_file_names, our_file_names, groups)


def conf_interval(data):
    n = len(data)
    stderr = sem(data)
    h = stderr * t.ppf((1 + 0.95) / 2, n - 1)
    return h


differences = []
conf_intervals = []
all_scores = []

high_sig = 0
low_sig1 = 0
low_sig2 = 0


def compute_differences_and_plot(results, group1, group2):
    global high_sig
    global low_sig1
    global low_sig2

    metrics = ['rouge_scores1_base', 'rouge_scores1_our',
               'rouge_scores2_base', 'rouge_scores2_our',
               'rouge_scoresl_base', 'rouge_scoresl_our',
               'recall_scores_base', 'recall_scores_our',
               'precision_scores_base', 'precision_scores_our',
               'f1_scores_base', 'f1_scores_our']  # 1 2  代表baseline 和our
    all_scores = []
    all_fig2_scores1 = []
    all_fig2_scores2 = []
    for metric in metrics:
        group1_scores = np.array(results[group1][metric])
        group2_scores = np.array(results[group2][metric])
        diff = np.abs(group1_scores - group2_scores)

        # diff = group1_scores - group2_scores
        all_scores.append(diff)
        all_fig2_scores1.append(np.array(results[group1][metric]))
        all_fig2_scores2.append(np.array(results[group2][metric]))

    # Plot setup
    age_categories = ['Baseline', 'Proposed']
    colors = ['#74a9cf', '#0570b0']

    plt.figure(figsize=(6, 3.8))
    bar_width = 0.35
    space_between_groups = 0.3
    for_calculation = []
    for i in range(6):  # Assuming the first 6 metrics are the ones we plot
        for j, cat in enumerate(age_categories):
            data = all_scores[i * 2 + j]
            mean = np.mean(data)
            for_calculation.append(mean)
            semm = stats.sem(data)
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=semm)
            bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
            plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]],
                    capsize=5)
        stat, p_value = stats.mannwhitneyu(all_scores[i * 2], all_scores[i * 2 + 1], alternative='greater')
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.1:
            stars = '*'
        else:
            stars = ''
        if stars:  # Check if the results are statistically significant
            max_mean = max(np.mean(all_scores[i * 2]), np.mean(all_scores[i * 2 + 1]))
            plt.text(i * (len(age_categories) * bar_width + space_between_groups) + bar_width / 2, max_mean + 0.01, stars,
                     fontsize=15, ha='center')
            high_sig += 1

    # X-axis and labels
    xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (
            len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(6)]
    xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
    plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=15)
    plt.ylabel('MFD', fontsize=15)
    plt.ylim(0, 0.08)
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               [cat.capitalize() for cat in age_categories], fontsize=12, loc='upper left')
    plt.title(f'{group1.capitalize()} vs {group2.capitalize()}')
    plt.tight_layout()
    filename = f'blue_{group1}_vs_{group2}.pdf'
    plt.savefig(filename)

    # 绘制柱状图
    plt.figure(figsize=(6, 3.8))
    colors = ['#b9d9b7', '#7da97a']
    for_calculation = []

    to_check = group1
    for i in range(6):  # Assuming the first 6 metrics are the ones we plot
        for j, cat in enumerate(age_categories):
            data = all_fig2_scores1[i * 2 + j]
            mean = np.mean(data)

            for_calculation.append(mean)

            semm = stats.sem(data)
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=semm)
            bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
            plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]],
                    capsize=5)
        stat, p_value = stats.mannwhitneyu(all_fig2_scores1[i * 2], all_fig2_scores1[i * 2 + 1], alternative='greater')
        if p_value < 0.05:  # Check if the results are statistically significant
            max_mean = max(np.mean(all_fig2_scores1[i * 2]), np.mean(all_fig2_scores1[i * 2 + 1]))
            plt.text(i * (len(age_categories) * bar_width + space_between_groups) + bar_width / 2, max_mean + 0.01, '*',
                     fontsize=15, ha='center')
            low_sig1 += 1
    xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (
            len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(6)]
    xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
    plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=15)
    plt.ylabel('Score', fontsize=15)
    plt.title(f'{to_check.capitalize()}')
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               [cat.capitalize() for cat in age_categories], fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'green_{to_check}.pdf')

    # 绘制柱状图
    plt.figure(figsize=(6, 3.8))
    colors = ['#b9d9b7', '#7da97a']
    for_calculation = []

    to_check = group2
    for i in range(6):  # Assuming the first 6 metrics are the ones we plot
        for j, cat in enumerate(age_categories):
            data = all_fig2_scores2[i * 2 + j]
            mean = np.mean(data)

            for_calculation.append(mean)

            semm = stats.sem(data)
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=semm)
            bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
            plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]],
                    capsize=5)
        stat, p_value = stats.mannwhitneyu(all_fig2_scores2[i * 2], all_fig2_scores2[i * 2 + 1], alternative='greater')
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = ''
        if stars:  # Check if the results are statistically significant
            max_mean = max(np.mean(all_fig2_scores1[i * 2]), np.mean(all_fig2_scores1[i * 2 + 1]))
            plt.text(i * (len(age_categories) * bar_width + space_between_groups) + bar_width / 2, max_mean + 0.01, stars,
                     fontsize=15, ha='center')
            low_sig2 += 1
    xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (
            len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(6)]
    xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
    plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=15)
    plt.ylabel('Score', fontsize=15)
    plt.title(f'{to_check.capitalize()}')
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               [cat.capitalize() for cat in age_categories], fontsize=12, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'green_{to_check}.pdf')


group_pairs = [('Young', 'Aged'), ('Female', 'Male'), ('White', 'Black')]
for i, (check1, check2) in enumerate(group_pairs):
    compute_differences_and_plot(results, check1, check2)
print(f'high sig {high_sig}')
print(f'low sig1 {low_sig1}')
print(f'low sig2 {low_sig2}')
