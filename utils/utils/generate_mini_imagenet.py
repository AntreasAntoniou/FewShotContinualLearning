import os
from shutil import copyfile
import tqdm
import sys

source_image_dir = '/home/antreas/datasets/mini_imagenet/images'
target_image_dir = '/home/antreas/datasets/mini_imagenet_full_size/'
csv_split_dir = '/home/antreas/HowToTrainYourMAMLPytorch_research_edition/utils/mini_imagenet_splits'


def load_csv_file_into_dict(csv_filepath):
    with open(csv_filepath, mode='r') as open_csv:
        data_lines = open_csv.readlines()

    image_to_label = dict()
    label_to_images = dict()

    for line in data_lines[1:]:
        image_filename, class_imagenet_id = line.replace("\n", "").split(",")
        if class_imagenet_id in label_to_images:
            label_to_images[class_imagenet_id].append(image_filename)
        else:
            label_to_images[class_imagenet_id] = [image_filename]

        image_to_label[image_filename] = class_imagenet_id

    return label_to_images, image_to_label


def load_set_to_images_dict(csv_split_source_dir):
    set_to_image_to_label = dict()
    set_to_label_to_images = dict()

    for subdir, dir, files in os.walk(csv_split_source_dir):
        for file in files:
            if file.endswith(".csv"):
                filepath = os.path.join(subdir, file)
                set_to_label_to_images[file.replace(".csv", "")], set_to_image_to_label[
                    file.replace(".csv", "")] = load_csv_file_into_dict(filepath)

    return set_to_label_to_images, set_to_image_to_label


def save_images_into_dir_structure(csv_split_dir, source_image_dir, target_image_dir):
    label_to_images, image_to_label = load_set_to_images_dict(csv_split_source_dir=csv_split_dir)

    counts = 0
    for subdir, dir, files in os.walk(source_image_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                counts += 1

    with tqdm.tqdm(total=counts) as pbar_copy:
        for set_name in image_to_label.keys():
            for image_name in image_to_label[set_name].keys():
                label = image_to_label[set_name][image_name]
                source_file_dir = os.path.join(source_image_dir, image_name)
                target_dir = os.path.join(target_image_dir, set_name, label)
                target_file_dir = os.path.join(target_dir, image_name)

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)

                copyfile(source_file_dir, target_file_dir)

                pbar_copy.update(1)


save_images_into_dir_structure(csv_split_dir=csv_split_dir, source_image_dir=source_image_dir,
                               target_image_dir=target_image_dir)
