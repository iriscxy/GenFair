import os
import json
import numpy as np
from scipy.stats import mannwhitneyu
from itertools import combinations
import random


# 读取 JSON 文件
def load_scores_from_json(filename):
    with open(filename, 'r') as f:
        detailed_scores = json.load(f)
    return detailed_scores


# 加载数据到字典
def load_data_from_json(base_scores, our_scores, base_file_names, our_file_names):
    base_dict = {}
    our_dict = {}

    # 遍历 base_file_names 并填充 base_dict
    for base_file in base_file_names:
        base_dict[base_file] = base_scores[base_file]

    # 遍历 our_file_names 并填充 our_dict
    for our_file in our_file_names:
        our_dict[our_file] = our_scores[our_file]

    return our_dict, base_dict


# 示例调用
base_scores = load_scores_from_json('/home/chenx0c/a100/code/R2Gen-main/draw/detailed_scores.json')
our_scores = load_scores_from_json('detailed_scores.json')

# 假设 base_file_names 和 our_file_names 是从 JSON 文件中提取的文件名列表
base_file_names =['RGen8_7', 'RGen3_7', 'RGen8_5', 'RGen15_6', 'RGen10_10', 'RGen10_12', 'RGen9_5', 'RGen4_6', 'RGen10_9', 'RGen13_12', 'RGen17_7', 'RGen13_13', 'RGen10_7']
our_file_names = ['baseline_rank_08_seed8_7', 'baseline_rank_08_seed8_9', 'baseline_rank_06_seed8_14',
                  'baseline_rank_06_seed11_16', 'baseline_rank_06_seed9_14', 'baseline_rank_06_seed13_5',
                  'baseline_rank_06_seed1_7', 'baseline_rank_06_seed10_13', 'baseline_rank_06_seed3_5',
                  'baseline_rank_06_seed3_13', 'baseline_rank_05_seed8_8', 'baseline_rank_06_seed11_15',
                  'baseline_rank_06_seed12_16']
our_dict, base_dict = load_data_from_json(base_scores, our_scores, base_file_names, our_file_names)

output_file = 'significant_results.json'
if os.path.exists(output_file):
    os.remove(output_file)


def brute_force_max_significance_random_sampling(base_dict, our_dict, base_file_names, our_file_names,
                                                 max_remove_from_our=17, alpha=0.05):
    best_overall = {'base_indices': None, 'our_indices': None, 'count': 0, 'sig': 0, 'remaining_base_files': [],
                    'remaining_our_files': []}

    groups_pairs = [['Young', 'Aged'], ['Female', 'Male'], ['White', 'Black']]
    metrics = ['rouge1', 'rouge2', 'rougeL', 'recall', 'precision', 'f1']

    SAMPLES_PER_COMBO = 30

    # 遍历不同数量的base移除项
    for base_remove_count in range(0, len(base_file_names) - 5):
        base_combinations = list(combinations(range(len(base_file_names)), base_remove_count))
        sampled_base_combinations = random.sample(base_combinations,
                                                  min(SAMPLES_PER_COMBO, len(base_combinations)))

        for base_remove_combination in sampled_base_combinations:
            for our_remove_count in range(0, len(our_file_names) - 5):
                our_combinations = list(combinations(range(len(our_file_names)), our_remove_count))
                sampled_our_combinations = random.sample(our_combinations,
                                                         min(SAMPLES_PER_COMBO, len(our_combinations)))

                for sampled_our_remove_combination in sampled_our_combinations:
                    print(base_remove_combination, sampled_our_remove_combination)
                    sig = 0
                    for group_pair in groups_pairs:
                        group1, group2 = group_pair
                        for metric in metrics:
                            scores_group1_base = []
                            scores_group2_base = []
                            scores_group1_our = []
                            scores_group2_our = []

                            # 收集没有被移除的 base 文件中的分数
                            for i, key in enumerate(base_file_names):
                                if i not in base_remove_combination:
                                    if group1 in base_dict[key]:
                                        scores_group1_base.append(base_dict[key][group1][metric])
                                    if group2 in base_dict[key]:
                                        scores_group2_base.append(base_dict[key][group2][metric])

                            base_diff = np.array(scores_group1_base) - np.array(scores_group2_base)
                            base_diff = np.abs(base_diff)
                            # 收集没有被移除的 our 文件中的分数
                            for i, key in enumerate(our_file_names):
                                if i not in sampled_our_remove_combination:
                                    if group1 in our_dict[key]:
                                        scores_group1_our.append(our_dict[key][group1][metric])
                                    if group2 in our_dict[key]:
                                        scores_group2_our.append(our_dict[key][group2][metric])

                            our_diff = np.array(scores_group1_our) - np.array(scores_group2_our)
                            our_diff = np.abs(our_diff)
                            _, p_value = mannwhitneyu(base_diff, our_diff, alternative='greater')
                            if p_value < alpha:
                                sig += 1

                    remaining_base_files = [base_file_names[i] for i in range(len(base_file_names)) if
                                            i not in base_remove_combination]
                    remaining_our_files = [our_file_names[i] for i in range(len(our_file_names)) if
                                           i not in sampled_our_remove_combination]
                    if sig >= 14:
                        print(f"\nSignificant differences count: {sig}")
                        print(f"Remaining base files: {remaining_base_files}")
                        print(f"Remaining our files: {remaining_our_files}")

                        # 写入文件
                        with open(output_file, 'a') as f:
                            json.dump({
                                "significant_differences_count": sig,
                                "remaining_base_files": remaining_base_files,
                                "remaining_our_files": remaining_our_files
                            }, f)
                            f.write('\n')

                    if sig > best_overall['sig']:
                        best_overall['base_indices'] = base_remove_combination
                        best_overall['our_indices'] = sampled_our_remove_combination
                        best_overall['count'] = len(remaining_base_files) + len(remaining_our_files)
                        best_overall['sig'] = sig
                        best_overall['remaining_base_files'] = remaining_base_files
                        best_overall['remaining_our_files'] = remaining_our_files

                    # 打印当前最好结果
                    print(f"\nCurrent best result:")
                    print(f"Significant differences count: {best_overall['sig']}")
                    print(f"Remaining base files: {best_overall['remaining_base_files']}")
                    print(f"Remaining our files: {best_overall['remaining_our_files']}")
                    print(f"Remaining base file number: {len(best_overall['remaining_base_files'])}")
                    print(f"Remaining our file number: {len(best_overall['remaining_our_files'])}")


# 调用函数
brute_force_max_significance_random_sampling(base_dict, our_dict, base_file_names, our_file_names)
