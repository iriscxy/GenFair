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

    for base_name, our_name in zip(base_file_names, our_file_names):
        data_dict[base_name] = {
            'base': base_scores.get(base_name, {}),
            'our': our_scores.get(our_name, {})
        }

    results = {group: {} for group in groups}

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

        for file_name, scores in data_dict.items():
            if group in scores['base']:
                base_scores_group = scores['base'][group]
                results[group]['rouge_scores1_base'].append(base_scores_group['rouge1'])
                results[group]['rouge_scores2_base'].append(base_scores_group['rouge2'])
                results[group]['rouge_scoresl_base'].append(base_scores_group['rougeL'])
                results[group]['recall_scores_base'].append(base_scores_group['recall'])
                results[group]['precision_scores_base'].append(base_scores_group['precision'])
                results[group]['f1_scores_base'].append(base_scores_group['f1'])

            if group in scores['our']:
                our_scores_group = scores['our'][group]
                results[group]['rouge_scores1_our'].append(our_scores_group['rouge1'])
                results[group]['rouge_scores2_our'].append(our_scores_group['rouge2'])
                results[group]['rouge_scoresl_our'].append(our_scores_group['rougeL'])
                results[group]['recall_scores_our'].append(our_scores_group['recall'])
                results[group]['precision_scores_our'].append(our_scores_group['precision'])
                results[group]['f1_scores_our'].append(our_scores_group['f1'])

    return results


# 示例调用
base_scores = load_scores_from_json('/home/chenx0c/a100/code/R2Gen-main/draw/detailed_scores.json')
our_scores = load_scores_from_json('detailed_scores.json')

# 假设 base_file_names 和 our_file_names 是从 JSON 文件中提取的文件名列表
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


def compute_differences_and_plot(results, group1, group2):
    metrics = ['rouge_scores1_base', 'rouge_scores1_our',
               'rouge_scores2_base', 'rouge_scores2_our',
               'rouge_scoresl_base', 'rouge_scoresl_our',
               'recall_scores_base', 'recall_scores_our',
               'precision_scores_base', 'precision_scores_our',
               'f1_scores_base', 'f1_scores_our']
    all_scores = []
    all_fig2_scores1 = []
    all_fig2_scores2 = []
    for metric in metrics:
        group1_scores = np.array(results[group1][metric])
        group2_scores = np.array(results[group2][metric])
        diff = np.abs(group1_scores - group2_scores)
        all_scores.append(diff)
        all_fig2_scores1.append(np.array(results[group1][metric]))
        all_fig2_scores2.append(np.array(results[group2][metric]))

    age_categories = ['Baseline', 'Proposed']
    colors = ['#74a9cf', '#0570b0']

    plt.figure(figsize=(6, 3.8))
    bar_width = 0.35
    space_between_groups = 0.3

    print(f"\nGroup: {group1} vs {group2}")
    for i in range(6):  # Assuming the first 6 metrics are the ones we plot
        for j, cat in enumerate(age_categories):
            data = all_scores[i * 2 + j]
            mean = np.mean(data)
            semm = stats.sem(data)
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=semm)
            bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
            plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]],
                    capsize=5)
            print(f"{cat} - {metrics[i * 2 + j]}: Mean = {mean:.3f}, SEM = {semm:.3f}, CI = ({ci[0]:.3f}, {ci[1]:.3f})")

        stat, p_value = stats.mannwhitneyu(all_scores[i * 2], all_scores[i * 2 + 1], alternative='greater')
        if p_value < 0.05:
            max_mean = max(np.mean(all_scores[i * 2]), np.mean(all_scores[i * 2 + 1]))
            plt.text(i * (len(age_categories) * bar_width + space_between_groups) + bar_width, max_mean + 0.01, '*',
                     fontsize=15, ha='center')

    xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (
            len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(6)]
    xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
    plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=10)
    plt.ylabel('MFD', fontsize=15)
    plt.ylim(0, 0.08)
    plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors],
               [cat.capitalize() for cat in age_categories], fontsize=12, loc='upper left')
    plt.title(f'{group1.capitalize()} vs {group2.capitalize()}')
    plt.tight_layout()
    plt.savefig(f'blue_{group1}_vs_{group2}.pdf')

    results_df = pd.DataFrame()
    print(f"\nGroup: {group1}")
    for i in range(6):
        for j, cat in enumerate(age_categories):
            data = all_fig2_scores1[i * 2 + j]
            mean = np.mean(data)
            semm = stats.sem(data)
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=semm)
            print(f"{cat} - {metrics[i * 2 + j]}: Mean = {mean:.3f}, SEM = {semm:.3f}, CI = ({ci[0]:.3f}, {ci[1]:.3f})")
            # 将原始数据添加到DataFrame
            temp_df = pd.DataFrame(data, columns=[f"{cat} - {metrics[i * 2 + j]}"])
            results_df = pd.concat([results_df, temp_df], axis=1)

    print(f"\nGroup: {group2}")
    for i in range(6):
        for j, cat in enumerate(age_categories):
            data = all_fig2_scores2[i * 2 + j]
            mean = np.mean(data)
            semm = stats.sem(data)
            ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=semm)
            print(f"{cat} - {metrics[i * 2 + j]}: Mean = {mean:.3f}, SEM = {semm:.3f}, CI = ({ci[0]:.3f}, {ci[1]:.3f})")
            # 将原始数据添加到DataFrame
            temp_df = pd.DataFrame(data, columns=[f"{cat} - {metrics[i * 2 + j]}"])
            results_df = pd.concat([results_df, temp_df], axis=1)
    return results_df
def compute_overall_statistics(results):
    metrics = ['rouge_scores1_base', 'rouge_scores1_our',
               'rouge_scores2_base', 'rouge_scores2_our',
               'rouge_scoresl_base', 'rouge_scoresl_our',
               'recall_scores_base', 'recall_scores_our',
               'precision_scores_base', 'precision_scores_our',
               'f1_scores_base', 'f1_scores_our']
    all_scores = {metric: [] for metric in metrics}
    for group, group_results in results.items():
        for metric in metrics:
            all_scores[metric].extend(group_results[metric])

    for metric, scores in all_scores.items():
        mean = np.mean(scores)
        semm = stats.sem(scores)
        ci = stats.t.interval(0.95, len(scores) - 1, loc=mean, scale=semm)
        print(f"{metric}: Mean = {mean:.3f}, SEM = {semm:.3f}, CI = ({ci[0]:.3f}, {ci[1]:.3f})")

all_results = []
group_pairs = [('Young', 'Aged'), ('Female', 'Male'), ('White', 'Black')]
for i, (check1, check2) in enumerate(group_pairs):
    result_dict=compute_differences_and_plot(results, check1, check2)
    all_results.append(result_dict)
    # 将所有结果合并到一个DataFrame中
    results_df = pd.DataFrame(all_results)

    # 将结果写入Excel文件
    results_df.to_excel("all_results.xlsx", index=False)
compute_overall_statistics(results)
