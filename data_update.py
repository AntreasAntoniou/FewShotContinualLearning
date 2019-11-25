import numpy as np
import torch
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.dataset_tools import get_label_set, load_dataset, load_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


def augment_image(image, transforms):
    for transform_current in transforms:
        image = transform_current(image)

    return image


class FewShotLearningDatasetParallel(Dataset):
    def __init__(self, dataset_path, dataset_name, indexes_of_folders_indicating_class, train_val_test_split,
                 labels_as_int, transforms, num_classes_per_set, num_samples_per_support_class,
                 num_samples_per_target_class, seed, sets_are_pre_split,
                 load_into_memory, set_name, num_tasks_per_epoch):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.indexes_of_folders_indicating_class = indexes_of_folders_indicating_class

        self.labels_as_int = labels_as_int
        self.train_val_test_split = train_val_test_split

        self.num_samples_per_support_class = num_samples_per_support_class
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_target_class = num_samples_per_target_class

        self.dataset = load_dataset(dataset_path, dataset_name, labels_as_int, seed, sets_are_pre_split,
                                    load_into_memory,
                                    indexes_of_folders_indicating_class, train_val_test_split)[set_name]

        self.num_tasks_per_epoch = num_tasks_per_epoch

        self.dataset_size_dict = {key: len(self.dataset[key]) for key in list(self.dataset.keys())}

        self.index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(dataset_path, dataset_name)
        self.label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(dataset_path, dataset_name)

        self.label_set = get_label_set(index_to_label_name_dict_file=self.index_to_label_name_dict_file)
        self.data_length = np.sum([len(self.dataset[key]) for key in self.dataset])

        self.seed = seed
        self.transforms = transforms

        print("data", self.data_length)

    def get_set(self, seed):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """

        rng = np.random.RandomState(seed)
        selected_classes = rng.choice(list(self.dataset_size_dict.keys()),
                                      size=self.num_classes_per_set, replace=False)
        rng.shuffle(selected_classes)

        episode_labels = [i for i in range(self.num_classes_per_set)]

        class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                  zip(selected_classes, episode_labels)}

        set_paths = [sample_path for
                     class_idx in selected_classes for sample_path in
                     rng.choice(self.dataset[class_idx],
                                size=self.num_samples_per_support_class + self.num_samples_per_target_class,
                                replace=False)]

        y = np.array([(self.num_samples_per_support_class + self.num_samples_per_target_class) * [
            class_to_episode_label[class_idx]]
                      for class_idx in selected_classes])

        x = torch.stack([augment_image(load_image(image_path), transforms=self.transforms) for image_path in set_paths])

        y = y.reshape(self.num_classes_per_set,
                      self.num_samples_per_support_class + self.num_samples_per_target_class)

        x = x.view(self.num_classes_per_set,
                   self.num_samples_per_support_class + self.num_samples_per_target_class, x.shape[1], x.shape[2],
                   x.shape[3])

        x_support_set = x[:, :self.num_samples_per_support_class]
        y_support_set = y[:, :self.num_samples_per_support_class]

        x_target_set = x[:, self.num_samples_per_support_class:]
        y_target_set = y[:, self.num_samples_per_support_class:]

        return x_support_set, x_target_set, y_support_set, y_target_set

    def __len__(self):
        return self.num_tasks_per_epoch

    def __getitem__(self, idx):
        return self.get_set(seed=self.seed + idx)


class FewShotContinualLearningDatasetParallel(Dataset):
    def __init__(self, dataset_path, dataset_name, indexes_of_folders_indicating_class, train_val_test_split,
                 labels_as_int, transforms, num_classes_per_set, num_continual_subtasks_per_task,
                 num_samples_per_support_class,
                 num_samples_per_target_class, seed, sets_are_pre_split,
                 load_into_memory, set_name, num_tasks_per_epoch, overwrite_classes_in_each_task):
        """
        A data provider class inheriting from Pytorch's Dataset class. It takes care of creating task sets for
        our few-shot learning model training and evaluation
        :param args: Arguments in the form of a Bunch object. Includes all hyperparameters necessary for the
        data-provider. For transparency and readability reasons to explicitly set as self.object_name all arguments
        required for the data provider, such that the reader knows exactly what is necessary for the data provider/
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.indexes_of_folders_indicating_class = indexes_of_folders_indicating_class

        self.labels_as_int = labels_as_int
        self.train_val_test_split = train_val_test_split

        self.num_samples_per_support_class = num_samples_per_support_class
        self.num_classes_per_set = num_classes_per_set
        self.num_samples_per_target_class = num_samples_per_target_class
        self.num_continual_subtasks_per_task = num_continual_subtasks_per_task
        self.overwrite_classes_in_each_task = overwrite_classes_in_each_task

        self.dataset = load_dataset(dataset_path, dataset_name, labels_as_int, seed, sets_are_pre_split,
                                    load_into_memory,
                                    indexes_of_folders_indicating_class, train_val_test_split)[set_name]

        self.num_tasks_per_epoch = num_tasks_per_epoch

        self.dataset_size_dict = {key: len(self.dataset[key]) for key in list(self.dataset.keys())}

        self.index_to_label_name_dict_file = "{}/map_to_label_name_{}.json".format(dataset_path, dataset_name)
        self.label_name_to_map_dict_file = "{}/label_name_to_map_{}.json".format(dataset_path, dataset_name)

        self.label_set = get_label_set(index_to_label_name_dict_file=self.index_to_label_name_dict_file)
        self.data_length = np.sum([len(self.dataset[key]) for key in self.dataset])

        self.seed = seed
        self.transforms = transforms

        print("data", self.data_length)

    def get_set(self, seed):
        """
        Generates a task-set to be used for training or evaluation
        :param set_name: The name of the set to use, e.g. "train", "val" etc.
        :return: A task-set containing an image and label support set, and an image and label target set.
        """
        # same num output classes

        rng = np.random.RandomState(seed)

        selected_classes = rng.choice(list(self.dataset_size_dict.keys()),
                                      size=self.num_classes_per_set * self.num_continual_subtasks_per_task,
                                      replace=False)

        rng.shuffle(selected_classes)

        if not self.overwrite_classes_in_each_task:
            episode_labels = [i for i in range(self.num_classes_per_set * self.num_continual_subtasks_per_task)]

            class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                      zip(selected_classes, episode_labels)}
        else:
            episode_labels = self.num_continual_subtasks_per_task * [i for i in range(self.num_classes_per_set)]

            class_to_episode_label = {selected_class: episode_label for (selected_class, episode_label) in
                                      zip(selected_classes, episode_labels)}

        set_paths = [sample_path for
                     class_idx in selected_classes for sample_path in
                     rng.choice(self.dataset[class_idx],
                                size=self.num_samples_per_support_class + self.num_samples_per_target_class,
                                replace=False)]

        y = np.array([(self.num_samples_per_support_class + self.num_samples_per_target_class) * [
            class_to_episode_label[class_idx]]
                      for class_idx in selected_classes])

        x = torch.stack([augment_image(load_image(image_path), transforms=self.transforms) for image_path in set_paths])

        y = y.reshape((self.num_continual_subtasks_per_task, self.num_classes_per_set,
                       self.num_samples_per_support_class + self.num_samples_per_target_class))

        x = x.view(self.num_continual_subtasks_per_task, self.num_classes_per_set,
                   self.num_samples_per_support_class + self.num_samples_per_target_class, x.shape[1], x.shape[2],
                   x.shape[3])

        x_support_set = x[:, :, :self.num_samples_per_support_class]
        y_support_set = y[:, :, :self.num_samples_per_support_class]

        x_target_set = x[:, :, self.num_samples_per_support_class:]
        y_target_set = y[:, :, self.num_samples_per_support_class:]

        return x_support_set, x_target_set, y_support_set, y_target_set

    def __len__(self):
        return self.num_tasks_per_epoch

    def __getitem__(self, idx):
        return self.get_set(seed=self.seed + idx)


transforms_to_use = [transforms.Resize(size=(84, 84)), transforms.ToTensor()]

data = FewShotContinualLearningDatasetParallel(dataset_path='/home/antreas/datasets/mini_imagenet_full_size',
                                               dataset_name='mini_imagenet_full_size',
                                               indexes_of_folders_indicating_class=[-3, -2],
                                               train_val_test_split=[0.8, 0.05, 0.15],
                                               labels_as_int=False, transforms=transforms_to_use, num_classes_per_set=5,
                                               num_samples_per_support_class=1,
                                               num_samples_per_target_class=15, seed=100, sets_are_pre_split=True,
                                               load_into_memory=True, set_name='train', num_tasks_per_epoch=10000,
                                               num_continual_subtasks_per_task=5, overwrite_classes_in_each_task=True)

dataloader = DataLoader(data, batch_size=1, num_workers=4)
import tqdm
import time

with tqdm.tqdm(total=len(dataloader)) as pbar:
    for idx, (x_support_set, x_target_set, y_support_set, y_target_set) in enumerate(dataloader):
        pbar.update(1)
        print(x_support_set.shape, torch.max(y_support_set))
        time.sleep(0.2)
