import os
import json

def update_sample_rate(labeled_file_path, original_file_path):
    with open(labeled_file_path, 'r') as f:
        labeled_data = json.load(f)

    with open(original_file_path, 'r') as f:
        original_data = json.load(f)

    sample_rate = int(original_data['sample_rate'])
    labeled_data['sample_rate'] = sample_rate

    with open(labeled_file_path, 'w') as f:
        json.dump(labeled_data, f, indent=4)

    print(f"Updated sample_rate in {labeled_file_path} to {sample_rate}")

def scan_and_update():
    labeled_dir = 'point_labelled_DB'
    original_dir = 'DB'

    for root, dirs, files in os.walk(labeled_dir):
        for file in files:
            if file.endswith('.json'):
                labeled_file_path = os.path.join(root, file)
                original_file_path = os.path.join(original_dir, os.path.relpath(labeled_file_path, labeled_dir))

                if os.path.exists(original_file_path):
                    update_sample_rate(labeled_file_path, original_file_path)
                else:
                    print(f"Corresponding original file not found for {labeled_file_path}")

if __name__ == '__main__':
    scan_and_update()