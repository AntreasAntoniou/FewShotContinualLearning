import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_update import FewShotContinualLearningDatasetParallel

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
