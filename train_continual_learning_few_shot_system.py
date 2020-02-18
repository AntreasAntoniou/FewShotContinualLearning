from torch.utils.data import DataLoader

from utils.parser_utils import get_args

args, device = get_args()

from utils.dataset_tools import check_download_dataset
from data import ConvertToThreeChannels, FewShotLearningDatasetParallel
from torchvision import transforms
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import *

# Combines the arguments, model, data and experiment builders to run an experiment

if args.classifier_type == 'maml++_high-end':
    model = EmbeddingMAMLFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'maml++_low-end':
    model = VGGMAMLFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-fine-tune-scratch':
    model = FineTuneFromScratchFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-fine-tune-pretrained':
    model = FineTuneFromPretrainedFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-matching_network':
    model = MatchingNetworkFewShotClassifier(**args.__dict__)
else:
    raise NotImplementedError

check_download_dataset(dataset_name=args.dataset_name)

if args.image_channels == 3:
    transforms = [transforms.Resize(size=(args.image_height, args.image_width)), transforms.ToTensor(),
                  ConvertToThreeChannels(),
                  transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
elif args.image_channels == 1:
    transforms = [transforms.Resize(size=(args.image_height, args.image_width)), transforms.ToTensor()]

train_setup_dict = dict(dataset_name=args.dataset_name,
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
                        num_channels=args.image_channels,
                        num_support_sets=args.num_support_sets,
                        overwrite_classes_in_each_task=args.overwrite_classes_in_each_task,
                        class_change_interval=args.class_change_interval)

val_setup_dict = dict(dataset_name=args.dataset_name,
                      indexes_of_folders_indicating_class=args.indexes_of_folders_indicating_class,
                      train_val_test_split=args.train_val_test_split,
                      labels_as_int=args.labels_as_int, transforms=transforms,
                      num_classes_per_set=args.num_classes_per_set,
                      num_samples_per_support_class=args.num_samples_per_support_class,
                      num_samples_per_target_class=args.num_samples_per_target_class,
                      seed=args.seed,
                      sets_are_pre_split=args.sets_are_pre_split,
                      load_into_memory=args.load_into_memory, set_name='val',
                      num_tasks_per_epoch=600 ,
                      num_channels=args.image_channels,
                      num_support_sets=args.num_support_sets,
                      overwrite_classes_in_each_task=args.overwrite_classes_in_each_task,
                      class_change_interval=args.class_change_interval)

test_setup_dict = dict(dataset_name=args.dataset_name,
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
                       num_channels=args.image_channels,
                       num_support_sets=args.num_support_sets,
                       overwrite_classes_in_each_task=args.overwrite_classes_in_each_task,
                       class_change_interval=args.class_change_interval)

train_data = FewShotLearningDatasetParallel(**train_setup_dict)

val_data = FewShotLearningDatasetParallel(**val_setup_dict)

test_data = FewShotLearningDatasetParallel(**test_setup_dict)

data_dict = {'train': DataLoader(train_data, batch_size=args.batch_size,
                                 num_workers=args.num_dataprovider_workers),
             'val': DataLoader(val_data, batch_size=args.batch_size,
                               num_workers=args.num_dataprovider_workers),
             'test': DataLoader(test_data, batch_size=args.batch_size,
                                num_workers=args.num_dataprovider_workers)}

maml_system = ExperimentBuilder(model=model, data_dict=data_dict, experiment_name=args.experiment_name,
                                continue_from_epoch=args.continue_from_epoch,
                                total_iter_per_epoch=args.total_iter_per_epoch,
                                num_evaluation_tasks=args.num_evaluation_tasks, total_epochs=args.total_epochs,
                                batch_size=args.batch_size, max_models_to_save=args.max_models_to_save,
                                evaluate_on_test_set_only=args.evaluate_on_test_set_only,
                                args=args)
maml_system.run_experiment()
