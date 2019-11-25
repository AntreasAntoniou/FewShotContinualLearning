from utils.parser_utils import get_args

args, device = get_args()

from utils.dataset_tools import check_download_dataset
from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system import *

# Combines the arguments, model, data and experiment builders to run an experiment

if args.classifier_type == 'densenet-embedding-based':
    model = EmbeddingMAMLFewShotClassifier(args=args, device=device,
                                     im_shape=(2, args.image_channels,
                                               args.image_height, args.image_width))
elif args.classifier_type == 'vgg-based':
    model = VGGMAMLFewShotClassifier(args=args, device=device,
                                               im_shape=(2, args.image_channels,
                                                         args.image_height, args.image_width))
else:
    raise NotImplementedError

check_download_dataset(args=args)
data = MetaLearningSystemDataLoader
maml_system = ExperimentBuilder(use_features_instead_of_images=False, model=model, data=data, args=args, device=device)
maml_system.run_experiment()
