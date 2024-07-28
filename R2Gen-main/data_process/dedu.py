import os
import json

def collect_image_paths(file_path, image_paths_set):
    with open(file_path, 'r') as f:
        for line in f:
            content = json.loads(line)
            image_paths = content.get('image_path', [])
            for image_path in image_paths:
                if image_path in image_paths_set:
                    print(f"Duplicate found: {image_path}")
                else:
                    image_paths_set.add(image_path)

train_file = '/ibex/ai/home/chenx0c/data/fair/train_image_path.json'  # Replace with your actual train file path
test_file = '/ibex/ai/home/chenx0c/data/fair/test_image_path.json'  # Replace with your actual test file path

image_paths_set = set()

collect_image_paths(train_file, image_paths_set)
collect_image_paths(test_file, image_paths_set)

print("Image path collection completed.")
