import os
import pdb

import pandas as pd
import json
import re

# Load CSV files
split_df = pd.read_csv('mimic-cxr-2.0.0-split.csv')
pdb.set_trace()
patients_df = pd.read_csv('patients.csv')  # Assuming 'subject_id' is the correct column name
admissions_df = pd.read_csv('admissions.csv').drop_duplicates(subset='subject_id')
merged_df = pd.merge(patients_df, admissions_df, on='subject_id')
merged_df = pd.merge(merged_df, split_df, on='subject_id')


# Function to extract text from report
def extract_report_text(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        if 'FINDINGS:' not in text or 'IMPRESSION:' not in text:
            return None, None

        try:
            parts = text.split('FINDINGS:')
            filter_part = parts[1].split('IMPRESSION:')
            findings = filter_part[0].strip().replace('\n', ' ')
            impression = filter_part[1].strip().replace('\n', ' ')
            # Remove leading "impression" or "impression:" regardless of case
            impression = re.sub(r'^\s*(impression:?\s*)', '', impression, flags=re.IGNORECASE)
        except:
            return None, None
        # Skip if impression or findings are too short
        if len(impression.split()) <= 2 and len(findings.split()) <= 10:
            return None, None

        return impression, findings


report_dir = '/home/chenx0c/a100/data/mimic-cxr-reports'  # Adjust with your actual reports directory path
image_dir = '/home/chenx0c/a100/data/fair_image'  # Adjust with your actual reports directory path
processed_files = set()  # Set to keep track of processed files
# Open output files in append mode
image_num = 0
with open('train_image_temp.json', 'w') as train_file, open('test_image_temp.json', 'w') as test_file,\
    open('valid_image_temp.json', 'w') as valid_file:
    for index, row in merged_df.iterrows():
        print(index)
        subject_id = row['subject_id']
        split_type = row['split']
        try:
            study_id = int(row['study_id'])  # Handle potential NaN values
        except ValueError:
            continue
        patient_folder = os.path.join(report_dir, f'p{str(subject_id)[:2]}', f'p{subject_id}')
        image_patient_folder = os.path.join(image_dir, f'p{str(subject_id)[:2]}', f'p{subject_id}')
        file_path = os.path.join(patient_folder, f's{study_id}.txt')
        image_path = os.path.join(image_patient_folder, f's{study_id}')
        files = [os.path.join(image_path, file) for file in os.listdir(image_path)]
        image_num += len(files)
        if os.path.exists(patient_folder) and file_path not in processed_files:
            impression, findings = extract_report_text(file_path)
            if impression is None:
                continue
            # Mark the file as processed
            processed_files.add(file_path)

            json_obj = {
                'study_id': study_id,
                'subject_id': subject_id,
                'impression': impression,
                'report': findings,
                'age': row.get('anchor_age', ''),
                'race': row.get('race', ''),
                'sex': row.get('gender', ''),
                'image_path':files
            }

            # Write directly to the appropriate file
            if split_type == 'train':
                json.dump(json_obj, train_file)
                train_file.write('\n')
            elif split_type == 'test':
                json.dump(json_obj, test_file)
                test_file.write('\n')
            elif split_type == 'validate':
                json.dump(json_obj, valid_file)
                valid_file.write('\n')
