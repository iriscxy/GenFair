import json
import pdb
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from evaluate import load
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

# Initialize ROUGE metric
rouge_metric = load('rouge')


# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'(?i)impression:', '', text)
    text = text.replace('\n', ' ').strip()
    text = " ".join(text.split())
    return text


# Function to read and categorize indices from JSON
def read_and_categorize_indices(label_file_path, combo):
    group_indices = {
        'Male': [],
        'Female': [],
        'White': [],
        'Black': [],
        'Young': [],
        'Aged': []
    }

    # 读取数据文件，并根据条件分类
    with open(label_file_path, 'r') as file:
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
            if label['age'] < 65:
                group_indices['Young'].append(idx)
            if label['age'] >= 65:
                group_indices['Aged'].append(idx)
    indices_sets = [set(group_indices[attribute.split('_')[0]]) & set(group_indices[attribute.split('_')[1]]) for
                    attribute in combo]
    return_dict = {combo[0]: list(indices_sets[0]),
                   combo[1]: list(indices_sets[1])
                   }
    return return_dict


# Function to evaluate ROUGE scores
def evaluate_rouge_scores(decoded_file_path, reference_file_path, group_indices):
    with open(decoded_file_path, 'r') as file:
        decs = [preprocess_text(line.strip()) for line in file]

    groups = {key: [] for key in group_indices.keys()}
    with open(reference_file_path, 'r') as file:
        references = [json.loads(line.strip()) for line in file]

    for category, indices in group_indices.items():
        for idx in indices:
            if idx < len(decs):
                groups[category].append((decs[idx], preprocess_text(references[idx]['findings'])))

    scores_dict = {}
    for group, texts in groups.items():
        if texts:
            predictions, references = zip(*texts)
            scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)
            scores_dict[group] = {
                'rouge1': scores['rouge1'],
                'rouge2': scores['rouge2'],
                'rougeL': scores['rougeL']
            }
    return scores_dict


# Function to get label accuracy
def get_label_accuracy(hypothesis, label_path, group_indices):
    reference = '/home/chenx0c/a100/code/chexpert-labeler/reference.csv'
    df_hyp_in = pd.read_csv(hypothesis)
    df_ref_in = pd.read_csv(reference)

    metrics = {}
    for group_name, indices in group_indices.items():
        if len(indices) == 0:
            continue
        df_hyp = df_hyp_in.iloc[indices]
        df_ref = df_ref_in.iloc[indices]

        df_hyp_pos1 = (df_hyp == 1).astype(int)
        del df_hyp_pos1["Reports"]
        df_hyp_pos1 = np.array(df_hyp_pos1)
        df_ref_pos1 = (df_ref == 1).astype(int)
        del df_ref_pos1["Reports"]
        df_ref_pos1 = np.array(df_ref_pos1)

        df_hyp_0 = (df_hyp == 0).astype(int)
        del df_hyp_0["Reports"]
        df_ref_0 = (df_ref == 0).astype(int)
        del df_ref_0["Reports"]

        df_hyp_neg1 = (df_hyp == -1).astype(int)
        del df_hyp_neg1["Reports"]
        df_ref_neg1 = (df_ref == -1).astype(int)
        del df_ref_neg1["Reports"]

        precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
        recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
        f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")

        metrics[group_name] = {
            'precision': precision_pos1,
            'recall': recall_pos1,
            'f1': f1_pos1
        }
    return metrics


ages = ['Aged', 'Young']
colors = ['White', 'Black']
genders = ['Female', 'Male']

# 第一步：从每对属性中生成所有可能的两两组合
first_combinations = [f"{age}_{color}" for age in ages for color in colors] + \
                     [f"{age}_{gender}" for age in ages for gender in genders] + \
                     [f"{color}_{gender}" for color in colors for gender in genders]


# 第二步：从生成的属性对中选择两个，形成一组新的组合，确保每个属性不重复
def validate_combination(comb):
    attributes = set()
    for item in comb:
        parts = item.split('_')
        attributes.update(parts)
    return len(attributes) == len(comb) * 2  # 保证没有重复属性


# 生成最终组合，每个组合为列表形式
final_combinations = [list(combo) for combo in itertools.combinations(first_combinations, 2) if
                      validate_combination(combo)]
# final_combinations = [['Aged_White', 'Aged_Black'], ['Young_White', 'Young_Black'], ['Aged_White', 'Young_Black'],
#                       ['Young_White', 'Aged_Black'],
#                       ['Female_White', 'Female_Black'], ['Male_White', 'Male_Black'], ['Female_White', 'Male_Black'],
#                       ['Female_Black', 'Male_White'],
#                       ['Aged_Female', 'Aged_Male'], ['Young_Female', 'Young_Male'], ['Aged_Female', 'Young_Male'],
#                       ['Aged_Male', 'Young_Female']]
final_combinations = [['Young_White', 'Young_Black']]
for combo in final_combinations:

    reference_file_path = '/home/chenx0c/a100/code/fair/test_label.json'
    group_indices = read_and_categorize_indices(reference_file_path, combo)

    directories_with_nums = {
        'baseline_rank_08_seed8': [7, 9],
        'baseline_rank_06_seed8': [14],
        'baseline_rank_06_seed9': [14],
        'baseline_rank_06_seed13': [5],
        'baseline_rank_06_seed10': [13],
        'baseline_rank_06_seed3': [5, 13],
        'baseline_rank_06_seed11': [15]
    }

    # File paths (use the paths from your previous sections)
    decoded_file_paths = [
        f'/home/chenx0c/a100/code/R2Gen-main_brio/results/{directory}/{num}/res.csv'
        for directory, nums in directories_with_nums.items()
        for num in nums
    ]

    chexpert_path1 = [
        f'/home/chenx0c/a100/code/chexpert-labeler/{directory}_{num}.csv'
        for directory, nums in directories_with_nums.items()
        for num in nums
    ]

    # Initialize scores for all groups
    all_rouge_scores = {}
    all_chexpert_scores = {}
    for combo_item in combo:
        all_rouge_scores[combo_item] = []
        all_chexpert_scores[combo_item] = []
    # Process each file and aggregate scores
    for decoded_file_path in decoded_file_paths:
        file_scores = evaluate_rouge_scores(decoded_file_path, reference_file_path, group_indices)
        for group_key in all_rouge_scores.keys():
            if group_key in file_scores:
                all_rouge_scores[group_key].append(file_scores[group_key])

    # Process CheXpert scores
    for hypothesis_file in chexpert_path1:
        file_scores = get_label_accuracy(hypothesis=hypothesis_file, label_path=reference_file_path,
                                         group_indices=group_indices)
        for group_key in all_chexpert_scores.keys():
            if group_key in file_scores:
                all_chexpert_scores[group_key].append(file_scores[group_key])


    # Function to combine scores and plot results
    def plot_scores_for_categories(age_categories, file_name):
        metrics = ['rouge1', 'rouge2', 'rougeL', 'recall', 'precision', 'f1']
        all_scores = {}

        new_age_categories = []
        for age_category in age_categories:
            category_key = age_category
            combined_scores = []
            new_age_categories.append(category_key)
            for rouge_score, chexpert_score in zip(all_rouge_scores[category_key], all_chexpert_scores[category_key]):
                combined_score = {**rouge_score, **chexpert_score}
                combined_scores.append(combined_score)
            all_scores[category_key] = combined_scores
        age_categories = new_age_categories
        plt.figure(figsize=(6, 3.8))
        colors = ['#f4d3a4', '#e67a4e']
        bar_width = 0.35
        space_between_groups = 0.3
        sig_num = 0
        for i, metric in enumerate(metrics):
            # if metric=='precision':
            #     pdb.set_trace()
            for j, cat in enumerate(age_categories):
                data = [score[metric] for score in all_scores[cat]]
                mean = np.mean(data)
                sem = stats.sem(data)
                ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)
                bar_position = i * (len(age_categories) * bar_width + space_between_groups) + j * bar_width
                plt.bar(bar_position, mean, width=bar_width, color=colors[j], yerr=[[mean - ci[0]], [ci[1] - mean]],
                        capsize=5)

            group1_data = [score[metric] for score in all_scores[age_categories[0]]]
            group2_data = [score[metric] for score in all_scores[age_categories[1]]]
            stat, p_value = stats.mannwhitneyu(group1_data, group2_data)
            if p_value < 0.001:
                stars = '***'
            elif p_value < 0.01:
                stars = '**'
            elif p_value < 0.05:
                stars = '*'
            else:
                stars = ''
            if stars:
                sig_num += 1
                max_mean = max(np.mean(group1_data), np.mean(group2_data))
                plt.text(i * (len(age_categories) * bar_width + space_between_groups) + bar_width / 2, max_mean + 0.01,
                         stars,
                         fontsize=15, ha='center')
        xticks = [(i * (len(age_categories) * bar_width + space_between_groups) + (
                len(age_categories) * bar_width) / 2 - bar_width / 2) for i in range(len(metrics))]
        xlabels = ['ROUGE1', 'ROUGE2', 'ROUGEL', 'Recall', 'Precision', 'F1']
        plt.xticks(ticks=xticks, labels=xlabels, fontsize=15, rotation=15)
        plt.ylabel('Score', fontsize=15)
        plt.ylim(0, 0.6)
        # plt.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors], age_categories, fontsize=12,
        #    loc='upper left')
        plt.title('Race & Age')

        plt.tight_layout()
        print(f'orange{file_name}_{sig_num}.pdf')
        plt.savefig(f'orange{file_name}_{sig_num}.pdf')


    name_str = '_'.join(combo)
    plot_scores_for_categories(combo, f'{name_str}')
