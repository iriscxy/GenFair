import json

def modify_image_paths(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line)
            if 'image_path' in data:
                data['image_path'] = [path.replace('/home/chenx0c/a100/data/fair_image/', '/ibex/ai/home/chenx0c/data/fair_image/') for path in data['image_path']]
            outfile.write(json.dumps(data) + '\n')

# 处理 train.json
modify_image_paths('/ibex/user/chenx0c/data/fair/train_label_candidate2.json', '/ibex/user/chenx0c/data/fair/train_label_candidate2_path.json')

# 处理 test.json