import json

# Paths to the input files with the new prefix
import pdb

test_json_path = '/ibex/user/chenx0c/data/fair/test_image_path.json'
train_json_path = '/ibex/user/chenx0c/data/fair/train_label_candidate2_path.json'

def read_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            content=json.loads(line)

            if 'candidates' not in content.keys():
                content['candidates']=['','']
            data.append(content)
    return data

# Load the data from the files
train_data = read_json_lines(train_json_path)
test_data = read_json_lines(test_json_path)

# Combine the data into a single dictionary
annotation_data = {
    'train': train_data,
    'test': test_data,
    'val': test_data  # Assuming 'val' is the same as 'test' for simplicity
}

# Write the combined data to annotation.json with the new prefix
output_path = '/ibex/user/chenx0c/data/fair/annotation_candidate2_path.json'
with open(output_path, 'w') as file:
    json.dump(annotation_data, file, indent=4)