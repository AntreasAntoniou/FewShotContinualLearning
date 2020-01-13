from torch.utils.data import DataLoader

from data_old import MetaLearningSystemDataLoader
from utils.parser_utils import get_args

args, device = get_args()

from utils.dataset_tools import check_download_dataset
from data import AddChannelsToTensor, FewShotLearningDatasetParallel
from torchvision import transforms
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import *

# Combines the arguments, model, data and experiment builders to run an experiment

if args.classifier_type == 'densenet-embedding-based':
    model = EmbeddingMAMLFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-based':
    model = VGGMAMLFewShotClassifier(**args.__dict__)
elif args.classifier_type == 'vgg-matching_network':
    model = MatchingNetworkFewShotClassifier(**args.__dict__)
else:
    raise NotImplementedError

check_download_dataset(args=args)


data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(use_features_instead_of_images=False, model=model, data=data, args=args, device=device)
maml_system.run_experiment()
