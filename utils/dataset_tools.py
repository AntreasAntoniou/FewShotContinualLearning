import concurrent.futures
import json
import os
import shutil

import numpy as np
import tqdm
from PIL import Image
from torchvision import transforms


def unzip_file(filepath_pack, filepath_to_store):
    print("unzipping", filepath_pack, "to", filepath_to_store)
    command_to_run = "tar -I pbzip2 -xf {} -C {}".format(filepath_pack, filepath_to_store)
    os.system(command_to_run)


def check_download_dataset(dataset_name):
    datasets = [dataset_name]
    dataset_paths = [os.path.join(os.path.abspath(os.environ['DATASET_DIR']), dataset_name)]

    done = False
    for dataset_idx, dataset_path in enumerate(dataset_paths):
        if dataset_path.endswith('/'):
            dataset_path = dataset_path[:-1]

        zip_directory = "{}.tar.bz2".format(os.path.join(os.environ['DATASET_DIR'], datasets[dataset_idx]))
        if not os.path.exists(zip_directory):
            print("New dataset not found, resetting")
            shutil.rmtree(dataset_path, ignore_errors=True)

        if not os.path.exists(os.environ['DATASET_DIR']):
            os.mkdir(os.environ['DATASET_DIR'])

        if not os.path.exists(dataset_path):
            print("Not found dataset folder structure.. searching for .tar.bz2 file")
            zip_directory = "{}.tar.bz2".format(os.path.join(os.environ['DATASET_DIR'], datasets[dataset_idx]))
            if not os.path.exists(zip_directory):
                print("Not found zip file, downloading..", zip_directory)
                return FileNotFoundError('Dataset is missing from the datasets folder, please download datasets and place '
                                         'them in the datasets folder as specified in the README.md file')

            else:
                print("Found zip file, unpacking")
            unzip_file(
                filepath_pack=os.path.join(os.environ['DATASET_DIR'], "{}.tar.bz2".format(datasets[dataset_idx])),
                filepath_to_store=os.environ['DATASET_DIR'])


        total_files = 0
        for subdir, dir, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg") or file.lower().endswith(
                        ".png") or file.lower().endswith(".pkl"):
                    total_files += 1
        print("count stuff________________________________________", dataset_path, total_files)
        if (total_files == 1623 * 20 and datasets[dataset_idx] == 'omniglot_dataset') or (
                total_files == 100 * 600 and 'mini_imagenet' in datasets[dataset_idx]) or (
                total_files == 11788 and "cub" in datasets[dataset_idx]) or (
                total_files == 779165 and "tiered_imagenet" in datasets[dataset_idx]) or (
                total_files == 200000 and "SlimageNet64" in datasets[dataset_idx]):
            print("file count is correct")
            done = True
        else:
            print("file count is wrong, redownloading dataset")
            return FileNotFoundError('Dataset file count is erroneous, please confirm that the dataset contains '
                                     'the right number of files, furthermore, confirm that utils/dataset_tools.py at'
                                     ' line 84-88 specifies the dataset you are using and its file count correctly')

        if not done:
            check_download_dataset(dataset_name, dataset_path)


def load_datapaths(dataset_dir, dataset_name, indexes_of_folders_indicating_class, labels_as_int):
    """
    If saved json dictionaries of the data are available, then this method loads the dictionaries such that the
    data is ready to be read. If the json dictionaries do not exist, then this method calls get_data_paths()
    which will build the json dictionary containing the class to filepath samples, and then store them.
    :return: data_image_paths: dict containing class to filepath list pairs.
             index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
             string-names of the class
             label_to_index: dictionary containing human understandable string mapped to numerical indexes
    """
    data_path_file = "{}/{}.json".format(dataset_dir, dataset_name)

    index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(dataset_dir, dataset_name)
    label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(dataset_dir, dataset_name)

    try:
        data_image_paths = load_from_json(filename=data_path_file)
        label_to_index = load_from_json(filename=label_name_to_map_dict_file)
        index_to_label_name_dict_file = load_from_json(filename=index_to_label_name_dict_file)
        return data_image_paths, index_to_label_name_dict_file, label_to_index
    except:
        print("Mapped data paths can't be found, remapping paths..")
        data_image_paths, code_to_label_name, label_name_to_code = get_data_paths(data_path=dataset_dir,
                                                                                  indexes_of_folders_indicating_class=indexes_of_folders_indicating_class,
                                                                                  labels_as_int=labels_as_int)
        save_to_json(dict_to_store=data_image_paths, filename=data_path_file)
        save_to_json(dict_to_store=code_to_label_name, filename=index_to_label_name_dict_file)
        save_to_json(dict_to_store=label_name_to_code, filename=label_name_to_map_dict_file)
        return load_datapaths(dataset_dir=dataset_dir, dataset_name=dataset_name,
                              indexes_of_folders_indicating_class=indexes_of_folders_indicating_class,
                              labels_as_int=labels_as_int)


def save_to_json(filename, dict_to_store):
    with open(os.path.abspath(filename), 'w') as f:
        json.dump(dict_to_store, fp=f)


def load_from_json(filename):
    with open(filename, mode="r") as f:
        load_dict = json.load(fp=f)

    return load_dict


def load_test_image(filepath):
    """
    Tests whether a target filepath contains an uncorrupted image. If image is corrupted, attempt to fix.
    :param filepath: Filepath of image to be tested
    :return: Return filepath of image if image exists and is uncorrupted (or attempt to fix has succeeded),
    else return None
    """
    image = None
    try:
        image = Image.open(filepath)
    except RuntimeWarning:
        os.system("convert {} -strip {}".format(filepath, filepath))
        print("converting")
        image = Image.open(filepath)
    except:
        print("Broken image")

    if image is not None:
        return filepath
    else:
        return None


def get_data_paths(data_path, labels_as_int, indexes_of_folders_indicating_class):
    """
    Method that scans the dataset directory and generates class to image-filepath list dictionaries.
    :return: data_image_paths: dict containing class to filepath list pairs.
             index_to_label_name_dict_file: dict containing numerical indexes mapped to the human understandable
             string-names of the class
             label_to_index: dictionary containing human understandable string mapped to numerical indexes
    """
    print("Get images from", data_path)
    data_image_path_list_raw = []
    labels = set()
    for subdir, dir, files in os.walk(data_path):
        for file in files:
            if (".jpeg") in file.lower() or (".png") in file.lower() or (".jpg") in file.lower():
                filepath = os.path.join(subdir, file)

                label = get_label_from_path(os.path.abspath(filepath),
                                            indexes_of_folders_indicating_class=indexes_of_folders_indicating_class,
                                            labels_as_int=labels_as_int)
                data_image_path_list_raw.append(filepath)

                labels.add(label)

    labels = sorted(labels)
    idx_to_label_name = {idx: label for idx, label in enumerate(labels)}
    label_name_to_idx = {label: idx for idx, label in enumerate(labels)}
    data_image_path_dict = {idx: [] for idx in list(idx_to_label_name.keys())}
    with tqdm.tqdm(total=len(data_image_path_list_raw)) as pbar_error:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            # Process the list of files, but split the work across the process pool to use all CPUs!
            for image_file in executor.map(load_test_image, (data_image_path_list_raw)):
                pbar_error.update(1)
                if image_file is not None:
                    label = get_label_from_path(image_file,
                                                indexes_of_folders_indicating_class=indexes_of_folders_indicating_class,
                                                labels_as_int=labels_as_int)
                    data_image_path_dict[label_name_to_idx[label]].append(image_file)

    return data_image_path_dict, idx_to_label_name, label_name_to_idx


def get_label_set(index_to_label_name_dict_file):
    """
    Generates a set containing all class numerical indexes
    :return: A set containing all class numerical indexes
    """
    index_to_label_name_dict_file = load_from_json(filename=index_to_label_name_dict_file)
    return set(list(index_to_label_name_dict_file.keys()))


def get_index_from_label(label, label_name_to_map_dict_file):
    """
    Given a class's (human understandable) string, returns the numerical index of that class
    :param label: A string of a human understandable class contained in the dataset
    :return: An int containing the numerical index of the given class-string
    """
    label_to_index = load_from_json(filename=label_name_to_map_dict_file)
    return label_to_index[label]


def get_label_from_index(index, index_to_label_name_dict):
    """
    Given an index return the human understandable label mapping to it.
    :param index: A numerical index (int)
    :return: A human understandable label (str)
    """
    return index_to_label_name_dict[index]


def get_label_from_path(filepath, indexes_of_folders_indicating_class, labels_as_int):
    """
    Given a path of an image generate the human understandable label for that image.
    :param filepath: The image's filepath
    :return: A human understandable label.
    """
    label_bits = filepath.split("/")
    label = "/".join([label_bits[idx] for idx in indexes_of_folders_indicating_class])

    if labels_as_int:
        label = int(label)

    return label


def load_image(image_path):
    """
    Given an image filepath and the number of channels to keep, load an image and keep the specified channels
    :param image_path: The image's filepath
    :param channels: The number of channels to keep
    :return: An image array of shape (h, w, channels), whose values range between 0.0 and 1.0.
    """
    try:
        image = Image.open(image_path)
        return image
    except:
        print(image_path, 'problem')
        return None


def load_batch(batch_image_paths):
    """
    Load a batch of images, given a list of filepaths
    :param batch_image_paths: A list of filepaths
    :return: A numpy array of images of shape batch, height, width, channels
    """

    image_batch = [load_image(image_path=image_path)
                   for image_path in batch_image_paths]

    return image_batch


def load_dataset(dataset_dir, dataset_name, labels_as_int, seed, sets_are_pre_split, load_into_memory,
                 indexes_of_folders_indicating_class, train_val_test_split):
    """
    Loads a dataset's dictionary files and splits the data according to the train_val_test_split variable stored
    in the args object.
    :return: Three sets, the training set, validation set and test sets (referred to as the meta-train,
    meta-val and meta-test in the paper)
    """
    rng = np.random.RandomState(seed=seed)

    if sets_are_pre_split == True:
        data_image_paths, index_to_label_name_dict, label_to_index = load_datapaths(dataset_dir=dataset_dir,
                                                                                    dataset_name=dataset_name,
                                                                                    indexes_of_folders_indicating_class=indexes_of_folders_indicating_class,
                                                                                    labels_as_int=labels_as_int)
        dataset_splits = dict()
        for key, value in data_image_paths.items():
            key = get_label_from_index(index=key, index_to_label_name_dict=index_to_label_name_dict)
            bits = key.split("/")

            set_name = bits[0]
            class_label = bits[1]
            if set_name not in dataset_splits:
                dataset_splits[set_name] = {class_label: value}
            else:
                dataset_splits[set_name][class_label] = value

    else:
        data_image_paths, index_to_label_name_dict_file, label_to_index = load_datapaths(dataset_dir=dataset_dir,
                                                                                         dataset_name=dataset_name,
                                                                                         indexes_of_folders_indicating_class=indexes_of_folders_indicating_class,
                                                                                         labels_as_int=labels_as_int)

        total_label_types = len(data_image_paths)
        num_classes_idx = np.arange(len(data_image_paths.keys()), dtype=np.int32)
        rng.shuffle(num_classes_idx)
        keys = list(data_image_paths.keys())
        values = list(data_image_paths.values())
        new_keys = [keys[idx] for idx in num_classes_idx]
        new_values = [values[idx] for idx in num_classes_idx]
        data_image_paths = dict(zip(new_keys, new_values))

        x_train_id, x_val_id, x_test_id = int(train_val_test_split[0] * total_label_types), \
                                          int(np.sum(train_val_test_split[:2]) * total_label_types), \
                                          int(total_label_types)

        x_train_classes = (class_key for class_key in list(data_image_paths.keys())[:x_train_id])
        x_val_classes = (class_key for class_key in list(data_image_paths.keys())[x_train_id:x_val_id])
        x_test_classes = (class_key for class_key in list(data_image_paths.keys())[x_val_id:x_test_id])
        x_train, x_val, x_test = {class_key: data_image_paths[class_key] for class_key in x_train_classes}, \
                                 {class_key: data_image_paths[class_key] for class_key in x_val_classes}, \
                                 {class_key: data_image_paths[class_key] for class_key in x_test_classes},
        dataset_splits = {"train": x_train, "val": x_val, "test": x_test}

    return dataset_splits
