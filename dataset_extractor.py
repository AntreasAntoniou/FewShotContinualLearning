import pickle
import os
from collections import defaultdict
import numpy as np
from PIL import Image
import shutil
import tqdm

source_path = "/home/antreas/datasets/imagenet_64x64/"
target_path = "/home/antreas/datasets/bold_imagenet/"
print('start')
if not os.path.exists(target_path):
    os.mkdir(target_path)


def unpickle(file):
    with open(file, 'rb') as fo:
        loaded = pickle.load(fo)
    return loaded


data_filepaths = []

for root, dirs, files in os.walk(source_path):
    for file in files:
        if not file.endswith(".zip"):
            filepath = os.path.join(root, file)
            data_filepaths.append(filepath)
            print(filepath)

print(data_filepaths)

count = defaultdict(lambda: 0)
with tqdm.tqdm(total=200000) as pbar:
    for file_path in data_filepaths:
        loaded_data = unpickle(file_path)

        data = loaded_data['data']

        data = np.reshape(data, newshape=(-1, 3, 64, 64)).transpose(0, 2, 3, 1)

        class_to_idx = dict()

        for idx, label in enumerate(loaded_data['labels']):
            if label in class_to_idx:
                class_to_idx[label].append(idx)
            else:
                class_to_idx[label] = [idx]

        for label_key in class_to_idx.keys():
            for idx in class_to_idx[label_key]:
                if count[label_key] <= 200:
                    image_arr = data[idx]

                    if label_key < 700:
                        split = 'train'
                    elif label_key < 800:
                        split = 'val'
                    else:
                        split = 'test'

                    filename = "n{0:04d}{1:04d}.png".format(label_key, count[label_key])
                    folder = os.path.join(target_path, split, "n{0:04d}".format(label_key))

                    im = Image.fromarray(image_arr)

                    if not os.path.exists(folder):
                        os.makedirs(folder)

                    filepath = os.path.join(folder, filename)
                    im.save(filepath)
                    count[label_key] += 1
                    pbar.update(1)
