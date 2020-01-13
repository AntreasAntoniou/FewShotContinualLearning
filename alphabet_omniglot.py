import os
import sys
import shutil

source_directory = '/home/antreas/datasets/omniglot_dataset'
target_directory = '/home/antreas/datasets/alphabet_omniglot_dataset'

if not os.path.exists(target_directory):
    os.makedirs(target_directory, exist_ok=True)

for subdir, dir, files in os.walk(source_directory):
    for file in files:
        if file.endswith('.png'):
            filepath = os.path.join(subdir, file)
            bits = filepath.split('/')
            del bits[-2]
            new_filepath = '/'.join(bits)
            new_filepath = new_filepath.replace(source_directory, target_directory)
            new_folder = '/'.join(new_filepath.split('/')[:-1])
            os.makedirs(new_folder, exist_ok=True)
            shutil.copy(filepath, new_filepath)