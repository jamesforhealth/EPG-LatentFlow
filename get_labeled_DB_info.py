import os
import json

def generate_directory_structure_json(labeled_db_path):
    directory_structure = {}

    for root, dirs, files in os.walk(labeled_db_path):
        for dir_name in dirs:
            subject_dir = os.path.join(root, dir_name)
            file_list = [f for f in os.listdir(subject_dir) if os.path.isfile(os.path.join(subject_dir, f))]
            directory_structure[dir_name] = file_list

    return directory_structure

def save_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    labeled_db_path = 'labeled_DB'  # 替換成你的資料夾路徑
    output_file = 'labeled_DB_info.json'

    directory_structure = generate_directory_structure_json(labeled_db_path)
    save_json(directory_structure, output_file)

    print(f"Directory structure saved to {output_file}")