import re
from collections import defaultdict

# Function to parse each line and extract scores
def parse_line(line):
    pattern = re.compile(r"Average scores for group (\w+_\w+): rouge1: ([\d.]+), rouge2: ([\d.]+), rougeL: ([\d.]+), recall: ([\d.]+), precision: ([\d.]+), f1: ([\d.]+)")
    match = pattern.match(line)
    if match:
        group = match.group(1)
        scores = {
            "rouge1": float(match.group(2)),
            "rouge2": float(match.group(3)),
            "rougeL": float(match.group(4)),
            "recall": float(match.group(5)),
            "precision": float(match.group(6)),
            "f1": float(match.group(7)),
        }
        return group, scores
    return None, None

# Function to read scores from a file and store them in a dictionary
def read_scores(file_path):
    scores_dict = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            group, scores = parse_line(line)
            if group:
                scores_dict[group].append(scores)
    return scores_dict

# Read scores from both files
file1_scores = read_scores('/home/chenx0c/a100/code/R2Gen-main/draw/inter_result.txt')
file2_scores = read_scores('/home/chenx0c/a100/code/R2Gen-main_brio/draw/inter_result.txt')

# Compare scores and print out the ones where file2 scores are higher than file1 for rouge1
for group in file2_scores:
    if group in file1_scores:
        for i, file2_score in enumerate(file2_scores[group]):
            if i < len(file1_scores[group]):
                file1_score = file1_scores[group][i]
                if file2_score["rouge1"] > file1_score["rouge1"]:
                    print(f"Group: {group}")
                    print(f"File 1 ROUGE1: {file1_score['rouge1']}")
                    print(f"File 2 ROUGE1: {file2_score['rouge1']}\n")
