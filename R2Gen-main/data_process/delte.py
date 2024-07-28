import os
import json

def delete_files(file_list):
    for file_path in file_list[1:]:  # Skip the first file
        if os.path.exists(file_path):
            os.remove(file_path)

def process_json_file(input_file):
    with open(input_file, 'r') as f:
        for line in f:
            content = json.loads(line)
            image_paths = content.get('image_path', [])
            if len(image_paths) > 1:
                delete_files(image_paths)  # Delete files except the first one

input_file = '/ibex/ai/home/chenx0c/data/fair/train_image_path.json'  # Replace with your actual input file path

process_json_file(input_file)
