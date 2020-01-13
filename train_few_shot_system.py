from utils.parser_utils import get_args

args, device = get_args()

from utils.dataset_tools import check_download_dataset
from data import FewShotLearningDatasetParallel, DataLoader
from torchvision import transforms
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import *

# Combines the arguments, model, data and experiment builders to run an experiment

if args.classifier_type == 'densenet-embedding-based':
    model = EmbeddingMAMLFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-based':
    model = VGGMAMLFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-matching_network':
    model = VGGMAMLFewShotClassifier(**args.__dict__)
else:
    raise NotImplementedError

check_download_dataset(args=args)

transforms = [transforms.Resize(size=(args.image_height, args.image_width)), transforms.ToTensor()]

train_data = FewShotLearningDatasetParallel(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                            indexes_of_folders_indicating_class=args.indexes_of_folders_indicating_class,
                                            train_val_test_split=args.train_val_test_split,
                                            labels_as_int=args.labels_as_int, transforms=transforms,
                                            num_classes_per_set=args.num_classes_per_set,
                                            num_samples_per_support_class=args.num_samples_per_support_class,
                                            num_samples_per_target_class=args.num_samples_per_target_class,
                                            seed=args.seed,
                                            sets_are_pre_split=args.sets_are_pre_split,
                                            load_into_memory=args.load_into_memory, set_name='train',
                                            num_tasks_per_epoch=args.total_epochs * args.total_iter_per_epoch,
                                            num_channels=args.image_channels)

val_data = FewShotLearningDatasetParallel(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                          indexes_of_folders_indicating_class=args.indexes_of_folders_indicating_class,
                                          train_val_test_split=args.train_val_test_split,
                                          labels_as_int=args.labels_as_int, transforms=transforms,
                                          num_classes_per_set=args.num_classes_per_set,
                                          num_samples_per_support_class=args.num_samples_per_support_class,
                                          num_samples_per_target_class=args.num_samples_per_target_class,
                                          seed=args.seed,
                                          sets_are_pre_split=args.sets_are_pre_split,
                                          load_into_memory=args.load_into_memory, set_name='val',
                                          num_tasks_per_epoch=600,
                                          num_channels=args.image_channels)

test_data = FewShotLearningDatasetParallel(dataset_path=args.dataset_path, dataset_name=args.dataset_name,
                                           indexes_of_folders_indicating_class=args.indexes_of_folders_indicating_class,
                                           train_val_test_split=args.train_val_test_split,
                                           labels_as_int=args.labels_as_int, transforms=transforms,
                                           num_classes_per_set=args.num_classes_per_set,
                                           num_samples_per_support_class=args.num_samples_per_support_class,
                                           num_samples_per_target_class=args.num_samples_per_target_class,
                                           seed=args.seed,
                                           sets_are_pre_split=args.sets_are_pre_split,
                                           load_into_memory=args.load_into_memory, set_name='test',
                                           num_tasks_per_epoch=600,
                                           num_channels=args.image_channels)

data_dict = {'train': DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_dataprovider_workers),
             'val': DataLoader(val_data, batch_size=args.batch_size, num_workers=args.num_dataprovider_workers),
             'test': DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_dataprovider_workers)}

maml_system = ExperimentBuilder(model=model, data_dict=data_dict, experiment_name=args.experiment_name,
                                continue_from_epoch=args.continue_from_epoch,
                                total_iter_per_epoch=args.total_iter_per_epoch,
                                num_evaluation_tasks=args.num_evaluation_tasks, total_epochs=args.total_epochs,
                                batch_size=args.batch_size, max_models_to_save=args.max_models_to_save,
                                evaluate_on_test_set_only=args.evaluate_on_test_set_only)
maml_system.run_experiment()
