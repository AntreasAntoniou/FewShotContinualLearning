# Task Specification

## Task Definition

A continual few-shot learning task consists of a number of (training) support sets and a single (evaluation) target set.

The support sets are controlled by the following hyperparameters:

1. Number of Support Sets (**NSS**): The number of support sets within a continual task.
2. Number of Classes Per Support Set (**N**): The number of classes within each support set.
3. Number of Samples Per Support Set Class (**K**): The number of data-points to be sampled for each class in the support set.
4. Number of Samples Per Target Set Class (**J**): The number of data-points to be sampled for each class in the target set. The number of classes in a target set is given by  <img src="https://render.githubusercontent.com/render/math?math=$\dfrac{NSS\times N}{CCI}$">
5. Class Change Interval (**CCI**): Assuming a task has NSS support sets, CCI defines the number of support sets 
sampled with the same class source, before that class source is resampled. This allows fine control of the type of task that will be generated.
6. Overwrite: A boolean variable that describes whether classes sampled at each support set will overwrite the class labels of the previously sampled support set (TRUE), or whether they will assigned new unique labels (FALSE).
For example, if support set 0 has class labels 0, 1, 2, 3, 4, then 
```
if Overwrite==FALSE: 
    support set 1 class labels: 5, 6, 7, 8, 9
elif Overwrite==TRUE:
    support set 1 class labels: 0, 1, 2, 3, 4
```
A task is sampled using the following algorithm:

<p align="center">
  <img width="460" height="300" src="https://github.com/AntreasAntoniou/FewShotContinualLearning/blob/master/materials/task_sampling_algorithm.png">
</p>


## Data Flow Dynamics

1. The model can only access one support set at a time for the purposes of knowledge extraction. 
Once this extraction is completed, the current support set is deleted. 
2. Task knowledge can be stored within a parameter vector/matrix or an embedding vector/matrix.
3. Once knowledge has been extracted by all the support sets, the model is tasked with predicting the classes of 
previously unseen samples. A generalization measure can be obtained by using the labels of this said, once the model
 has produced its predictions to compute a task-level generalization measure.
 
## Metrics

1. Test Generalization Performance: A proposed model should be evaluated on the test sets of Omniglot and SlimageNet, on all of the task
types of interest. This is done by presenting the model with a number of previously unseen continual tasks sampled from these test sets, and then
using the target set metrics as the task-level generalization metrics. To obtain a measure of generalization across the whole test set the model
should be evaluated on **600** unique tasks and then take the mean and standard deviation of both the accuracy and cross-entropy performance of the model.
These should be used as generalization measures to compare the model to other models.

2. Across-Task Memory (**ATM**): Since the knowledge storage vectors that store support set knowledge have unrestricted memory, we also incorporate
a metric that explicitly measures how memory efficient a certain model is, which can help differentiate between models of equal generalization performance.
This measure can be computed by <img src="https://render.githubusercontent.com/render/math?math=$\dfrac{M}{T}$"> 
where *M* is the total knowledge memory size across a whole task, and 
*T* is the total size of all samples in all the support sets within a task.  

3. Computational Cost/Inference Memory: This metric measures the computational expense of the joint learner + model operations, as well as the memory footprint. This memory footprint is different in ATM, as ATM measures how much information are kept from each task when the next task is observed, whereas the inference memory footprint measures the memory footprint that the model itself needs to execute one cycle of inference + learning. Both of these costs can be approximated as the total amount of MAC (Multiply Accumulate) operations that a given model executes. Computing the total number of MAC operations  

## Rules

1. A task with CCI=1, NSS=1 will generate the same task type no matter what Overwrite is set to. 
2. When classes are resampled for a new support set, assuming the CCI interval has been reached, the classes should be unique classes that have not appeared in any of the preceeding support sets within the current continual task.
3. When new samples are sampled, they should be unique samples, that have not been used in any other support sets of the current continual task.
4. For SlimageNet the splits should be the exact splits that appear within train, val, and test in https://zenodo.org/record/3672132.
5. The smaller the ATM of a given model the more memory efficient it is. Maximal efficiency is achieved at 0, where no memory is used, and minimum efficiency is reached at infinity, where the model has infinite memory. A model that can store the whole dataset it observed will have ATM=1, whereas a model that stores 10% of the information in a data-point will have an ATM=0.1.
