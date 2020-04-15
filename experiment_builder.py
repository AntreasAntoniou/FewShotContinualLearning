import os
import time

import numpy as np
import tqdm

from utils.storage import build_experiment_folder, save_statistics, save_to_json

class ExperimentBuilder(object):
    def __init__(self, data_dict, model, experiment_name, continue_from_epoch, max_models_to_save,
                 total_iter_per_epoch, total_epochs, num_evaluation_tasks, batch_size, evaluate_on_test_set_only,
                 args):
        """
        Initializes an experiment builder using a named tuple (args), a data provider (data), a meta learning system
        (model) and a device (e.g. gpu/cpu/n)
        :param args: A namedtuple containing all experiment hyperparameters
        :param data: A data provider of instance MetaLearningSystemDataLoader
        :param model: A meta learning system instance
        :param device: Device/s to use for the experiment
        """

        self.model = model
        self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
            experiment_name=experiment_name)

        self.total_losses = dict()
        self.state = dict()
        self.state['best_val_acc'] = 0.
        self.state['best_val_iter'] = 0
        self.state['current_iter'] = 0
        self.state['current_iter'] = 0
        self.start_epoch = 0
        self.max_models_to_save = max_models_to_save
        self.create_summary_csv = False
        self.evaluate_on_test_set_only = evaluate_on_test_set_only

        for key, value in args.__dict__.items():
            setattr(self, key, value)

        if continue_from_epoch == 'from_scratch':
            self.create_summary_csv = True

        elif continue_from_epoch == 'latest':
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_latest")
            print("attempting to find existing checkpoint", )
            if os.path.exists(checkpoint):
                try:
                    self.state = \
                        self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                              model_idx='latest')
                    self.start_epoch = int(self.state['current_iter'] / total_iter_per_epoch)
                except:
                    self.continue_from_epoch = 'from_scratch'
                    self.create_summary_csv = True

            else:
                self.continue_from_epoch = 'from_scratch'
                self.create_summary_csv = True
        elif int(continue_from_epoch) >= 0:
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_{}".format(continue_from_epoch))
            if os.path.exists(checkpoint):
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx=continue_from_epoch)
                self.start_epoch = int(self.state['current_iter'] / total_iter_per_epoch)
            else:
                self.continue_from_epoch = 'from_scratch'
                self.create_summary_csv = True

        self.data = data_dict
        self.total_iter_per_epoch = total_iter_per_epoch
        self.batch_size = batch_size
        self.total_epochs = total_epochs
        self.num_evaluation_tasks = num_evaluation_tasks

        print("train_seed {}, val_seed: {}, at start time".format(self.data["train"].dataset.seed,
                                                                  self.data["val"].dataset.seed))

        self.state['best_epoch'] = int(self.state['best_val_iter'] / total_iter_per_epoch)
        self.epoch = int(self.state['current_iter'] / total_iter_per_epoch)
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0
        print(self.state['current_iter'], int(total_iter_per_epoch * total_epochs))

    def build_summary_dict(self, total_losses, phase, summary_losses=None):
        """
        Builds/Updates a summary dict directly from the metric dict of the current iteration.
        :param total_losses: Current dict with total losses (not aggregations) from experiment
        :param phase: Current training phase
        :param summary_losses: Current summarised (aggregated/summarised) losses stats means, stdv etc.
        :return: A new summary dict with the updated summary statistics information.
        """
        if summary_losses is None:
            summary_losses = dict()

        for key in total_losses:
            if 'saved_logits' not in key:
                summary_losses["{}_{}_mean".format(phase, key)] = np.mean(total_losses[key])
                summary_losses["{}_{}_std".format(phase, key)] = np.std(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):
        """
        Builds a progress bar summary string given current summary losses dictionary
        :param summary_losses: Current summary statistics
        :return: A summary string ready to be shown to humans.
        """
        output_update = ""
        for key, value in zip(list(summary_losses.keys()), list(summary_losses.values())):
            if ("loss" in key or "accuracy" in key or 'opt' in key) and (not "pre" in key and not "post" in key):
                value = float(value)
                output_update += "{}: {:.4f}, ".format(key, value)

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(self, train_sample, sample_idx, epoch_idx, total_losses, current_iter, pbar_train):
        """
        Runs a training iteration, updates the progress bar and returns the total and current epoch train losses.
        :param train_sample: A sample from the data provider
        :param sample_idx: The index of the incoming sample, in relation to the current training run.
        :param epoch_idx: The epoch index.
        :param total_losses: The current total losses dictionary to be updated.
        :param current_iter: The current training iteration in relation to the whole experiment.
        :param pbar_train: The progress bar of the training.
        :return: Updates total_losses, train_losses, current_iter
        """

        losses, _ = self.model.run_train_iter(data_batch=train_sample, epoch=epoch_idx, current_iter=current_iter)

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if 'saved_logits' not in key:
                if key not in total_losses:
                    total_losses[key] = [float(value)]
                else:
                    total_losses[key].append(float(value))
            else:
                np.save(
                    os.path.join(self.samples_filepath, 'saved_{}_{}_logits'.format('train', current_iter)),
                    value.detach().cpu().numpy())

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(losses)

        pbar_train.update(1)
        pbar_train.set_description("training phase {} -> {}".format(self.epoch, train_output_update))

        current_iter += 1

        return train_losses, total_losses, current_iter

    def evaluation_iteration(self, val_sample, total_losses, pbar_val, phase):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_val: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """

        losses, _ = self.model.run_validation_iter(data_batch=val_sample)
        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase=phase)
        val_output_update = self.build_loss_summary_string(losses)

        pbar_val.update(1)
        pbar_val.set_description(
            "val_phase {} -> {}".format(self.epoch, val_output_update))

        return val_losses, total_losses

    def test_evaluation_iteration(self, val_sample, model_idx, sample_idx, per_model_per_batch_preds, pbar_test):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_test: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        losses, per_task_preds = self.model.run_validation_iter(data_batch=val_sample)

        per_model_per_batch_preds[model_idx].extend(list(per_task_preds))

        val_output_update = self.build_loss_summary_string(losses)

        pbar_test.update(1)
        pbar_test.set_description(
            "val_phase {} -> {}".format(self.epoch, val_output_update))

        return per_model_per_batch_preds

    def convert_into_continual_tasks(self, data_batch):

        x_support_set, x_target_set, y_support_set, y_target_set, x, y = data_batch

        return x_support_set, x_target_set, y_support_set, y_target_set,  x, y

    def save_models(self, model, epoch, state):
        """
        Saves two separate instances of the current model. One to be kept for history and reloading later and another
        one marked as "latest" to be used by the system for the next epoch training. Useful when the training/val
        process is interrupted or stopped. Leads to fault tolerant training and validation systems that can continue
        from where they left off before.
        :param model: Current meta learning model of any instance within the few_shot_learning_system.py
        :param epoch: Current epoch
        :param state: Current model and experiment state dict.
        """
        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_{}".format(int(epoch))),
                         state=state)

        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_latest"),
                         state=state)

        print("saved models to", self.saved_models_filepath)

    def pack_and_save_metrics(self, start_time, create_summary_csv, train_losses, val_losses, state):
        """
        Given current epochs start_time, train losses, val losses and whether to create a new stats csv file, pack stats
        and save into a statistics csv file. Return a new start time for the new epoch.
        :param start_time: The start time of the current epoch
        :param create_summary_csv: A boolean variable indicating whether to create a new statistics file or
        append results to existing one
        :param train_losses: A dictionary with the current train losses
        :param val_losses: A dictionary with the currrent val loss
        :return: The current time, to be used for the next epoch.
        """
        epoch_summary_losses = self.merge_two_dicts(first_dict=train_losses, second_dict=val_losses)

        if 'per_epoch_statistics' not in state:
            state['per_epoch_statistics'] = dict()

        for key, value in epoch_summary_losses.items():

            if key not in state['per_epoch_statistics']:
                state['per_epoch_statistics'][key] = [value]
            else:
                state['per_epoch_statistics'][key].append(value)

        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch
        epoch_summary_losses['epoch_run_time'] = time.time() - start_time

        if create_summary_csv:
            self.summary_statistics_filepath = save_statistics(self.logs_filepath, list(epoch_summary_losses.keys()),
                                                               create=True)
            self.create_summary_csv = False

        start_time = time.time()
        print("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

        self.summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                           list(epoch_summary_losses.values()))
        return start_time, state

    def evaluate_test_set_using_the_best_models(self, top_n_models):
        if 'per_epoch_statistics' in self.state:
            per_epoch_statistics = self.state['per_epoch_statistics']
            val_acc = np.copy(per_epoch_statistics['val_accuracy_mean'])
            val_idx = np.array([i for i in range(len(val_acc))])
            sorted_idx = np.argsort(val_acc, axis=0).astype(dtype=np.int32)[::-1][:top_n_models]

            sorted_val_acc = val_acc[sorted_idx]
            val_idx = val_idx[sorted_idx]
            print(sorted_idx)
            print(sorted_val_acc)

            top_n_idx = val_idx[:top_n_models]
            per_model_per_batch_preds = [[] for i in range(top_n_models)]
            per_model_per_batch_targets = [[] for i in range(top_n_models)]

            test_losses = [dict() for i in range(top_n_models)]
        else:
            top_n_idx = [i for i in range(top_n_models)]
            per_model_per_batch_preds = [[] for i in range(top_n_models)]
            per_model_per_batch_targets = [[] for i in range(top_n_models)]
            test_losses = [dict() for i in range(top_n_models)]

        for idx, model_idx in enumerate(top_n_idx):
            if 'per_epoch_statistics' in self.state:
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx=model_idx + 1)
            else:
                pass

            with tqdm.tqdm(total=int(self.num_evaluation_tasks / self.batch_size)) as pbar_test:
                for sample_idx, test_sample in enumerate(
                        self.data['test']):
                    test_sample = self.convert_into_continual_tasks(test_sample)
                    x_support_set, x_target_set, y_support_set, y_target_set, x, y = test_sample
                    per_model_per_batch_targets[idx].extend(np.array(y_target_set))
                    per_model_per_batch_preds = self.test_evaluation_iteration(val_sample=test_sample,
                                                                               sample_idx=sample_idx,
                                                                               model_idx=idx,
                                                                               per_model_per_batch_preds=per_model_per_batch_preds,
                                                                               pbar_test=pbar_test)
        per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)

        per_batch_max = np.argmax(per_batch_preds, axis=2)
        per_batch_targets = np.array(per_model_per_batch_targets[0]).reshape(per_batch_max.shape)

        accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
        accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

        test_losses = {"test_accuracy_mean": accuracy, "test_accuracy_std": accuracy_std}

        _ = save_statistics(self.logs_filepath,
                            list(test_losses.keys()),
                            create=True, filename="test_summary.csv")

        summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                      list(test_losses.values()),
                                                      create=False, filename="test_summary.csv")
        print(test_losses)
        print("saved test performance at", summary_statistics_filepath)

    def run_experiment(self):
        """
        Runs a full training experiment with evaluations of the model on the val set at every epoch. Furthermore,
        will return the test set evaluation results on the best performing validation model.
        """
        with tqdm.tqdm(initial=self.state['current_iter'],
                       total=int(self.total_iter_per_epoch * self.total_epochs)) as pbar_train:

            self.data['train'].dataset.set_current_iter_idx(self.state['current_iter'])

            while (self.state['current_iter'] < (self.total_epochs * self.total_iter_per_epoch)) and (
                    self.evaluate_on_test_set_only == False):

                for idx, train_sample in enumerate(self.data['train']):
                    train_sample = self.convert_into_continual_tasks(train_sample)

                    train_losses, total_losses, self.state['current_iter'] = self.train_iteration(
                        train_sample=train_sample,
                        total_losses=self.total_losses,
                        epoch_idx=(self.state['current_iter'] /
                                   self.total_iter_per_epoch),
                        pbar_train=pbar_train,
                        current_iter=self.state['current_iter'],
                        sample_idx=self.state['current_iter'])

                    better_val_model = False
                    if self.state['current_iter'] % self.total_iter_per_epoch == 0:

                        total_losses = dict()
                        val_losses = dict()
                        with tqdm.tqdm(total=len(self.data['val'])) as pbar_val:
                            for val_sample_idx, val_sample in enumerate(
                                    self.data['val']):

                                val_sample = self.convert_into_continual_tasks(val_sample)
                                val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                                     total_losses=total_losses,
                                                                                     pbar_val=pbar_val, phase='val')

                            if val_losses["val_accuracy_mean"] > self.state['best_val_acc']:
                                print("Best validation accuracy", val_losses["val_accuracy_mean"])
                                self.state['best_val_acc'] = val_losses["val_accuracy_mean"]
                                self.state['best_val_iter'] = self.state['current_iter']
                                self.state['best_epoch'] = int(
                                    self.state['best_val_iter'] / self.total_iter_per_epoch)

                        self.epoch += 1
                        self.state = self.merge_two_dicts(first_dict=self.merge_two_dicts(first_dict=self.state,
                                                                                          second_dict=train_losses),
                                                          second_dict=val_losses)



                        self.start_time, self.state = self.pack_and_save_metrics(start_time=self.start_time,
                                                                                 create_summary_csv=self.create_summary_csv,
                                                                                 train_losses=train_losses,
                                                                                 val_losses=val_losses,
                                                                                 state=self.state)
                        self.save_models(model=self.model, epoch=self.epoch, state=self.state)

                        self.total_losses = dict()

                        self.epochs_done_in_this_run += 1
                        # print(self.state['per_epoch_statistics']['val_accuracy_mean'])
                        save_to_json(filename=os.path.join(self.logs_filepath, "summary_statistics.json"),
                                     dict_to_store=self.state['per_epoch_statistics'])

            self.evaluate_test_set_using_the_best_models(top_n_models=5)
