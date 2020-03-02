# Task Specification

A continual few-shot learning task consists of a number of (training) support sets and a single (evaluation) target set.

The support sets are controlled by **five** hyperparameters:

1. Number of Support Sets (**NSS**): The number of support sets within a continual task.
2. Number of Classes Per Support Set (**N**): The number of classes within each support set.
3. Number of Samples Per Support Set Class (**K**): The number of data-points to be sampled for each class in the support set.
4. Number of Samples Per Target Set Class (**J**): The number of data-points to be sampled for each class in the target set. The number of classes in a target set is given by
5. Class Change Interval (**CCI**): Assuming a task has NSS support sets, CCI defines the number of support sets 
sampled with the same class source, before that class source is resampled. This allows fine control of the type of task that will be generated.

