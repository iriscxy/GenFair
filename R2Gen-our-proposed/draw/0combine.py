import json

# 读取 JSON 文件
def read_json(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# 读取 detailed_scores1.json
data1 = read_json('detailed_scores1.json')

# 读取 detailed_scores2.json
data2 = read_json('detailed_scores2.json')

# 读取 detailed_scores3.json
data3 = read_json('detailed_scores3.json')

# 合并三个字典
merged_data = {**data1, **data2, **data3}

# 将合并后的数据写入新文件 detailed_scores.json
with open('detailed_scores.json', 'w') as file:
    json.dump(merged_data, file, indent=4)

print("合并后的数据已写入 detailed_scores.json 文件。")