class MAMLPlusPlusFewShotClassifier(MAMLFewShotClassifier):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLPlusPlusFewShotClassifier, self).__init__(args=args, im_shape=im_shape, device=device)

    def build_module(self):

        self.classifier = VGGActivationNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                   num_classes_per_set, num_stages=4,
                                                   args=self.args, device=self.device, meta_classifier=True)

        print("init learning rate", self.args.init_learning_rate)
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=self.device,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    learnable_learning_rates=self.args.learnable_learning_rates,
                                                                    init_learning_rate=self.args.init_learning_rate)
        self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)

        task_name_params = self.get_inner_loop_parameter_dict(self.named_parameters())

        x_dummy = torch.zeros(self.im_shape)
        _, x_features = self.classifier.forward(x=x_dummy, num_step=0, training=True, output_features=True)

        print("Inner Loop parameters")
        for key, param in task_name_params.items():
            print(key, param.shape)

        self.use_cuda = self.args.use_cuda
        self.device = self.device
        self.started_running = False
        self.exclude_list = None if "none" in self.args.exclude_param_string else self.args.exclude_param_string
        exclude_param_string = None if "none" in self.args.exclude_param_string else self.args.exclude_param_string
        self.optimizer = optim.Adam(self.trainable_parameters(exclude_params_with_string=exclude_param_string),
                                    lr=self.args.meta_learning_rate,
                                    weight_decay=self.args.weight_decay, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                              T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)
        self.to(self.device)

        print("Outer Loop parameters")
        for name, param in self.trainable_names_parameters(exclude_params_with_string=exclude_param_string):
            if param.requires_grad:
                print(name, param.shape)

    def get_params_that_include_strings(self, included_strings, include_all=False):
        for name, param in self.named_parameters():
            if any([included_string in name for included_string in included_strings]) and not include_all:
                yield param
            if all([included_string in name for included_string in included_strings]) and include_all:
                yield param

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = self.args.minimum_per_task_contribution / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        self.classifier.zero_grad(params=names_weights_copy)
        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     num_step=current_step_idx)

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses_dict):
        losses = dict()
        total_losses = torch.stack([torch.stack(step_loss) for step_loss in total_losses_dict['total_per_step_losses']])
        per_step_loss = torch.mean(total_losses, dim=1)

        logging.debug(per_step_loss)

        losses['loss'] = per_step_loss[-1]

        for key, value in total_losses_dict.items():
            if type(value) == list:
                value = torch.Tensor(value)

            losses[key] = torch.mean(value)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set, y_support_set_original, y_target_set_original = data_batch

        num_metrics_to_track = 1
        metrics_dict = dict()
        metrics_dict['total_per_step_losses'] = [[] for i in range(num_metrics_to_track)]
        metrics_dict['accuracy'] = [[] for i in range(num_metrics_to_track)]
        metrics_dict['per_task_preds'] = [[] for i in range(len(x_target_set))]

        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):

            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            c, h, w = x_target_set_task.shape[-3:]

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            if self.started_running == False:
                print(x_support_set_task.shape, x_target_set_task.shape)
                self.started_running = True

            for num_step in range(num_steps):
                support_preds = self.classifier.forward(x=x_support_set_task, params=names_weights_copy,
                                                        training=True,
                                                        backup_running_statistics=True if (num_step == 0) else False,
                                                        num_step=num_step)
                support_loss = F.cross_entropy(support_preds, y_support_set_task)

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step,
                                                                  )

            pre_target_preds = self.classifier.forward(x=x_target_set_task, params=names_weights_copy,
                                                       training=True,
                                                       backup_running_statistics=False,
                                                       num_step=num_step)

            pre_target_loss = F.cross_entropy(pre_target_preds, y_target_set_task)

            metrics_dict['per_task_preds'][task_id] = target_preds.detach().cpu().numpy()
            softmax_target_preds = F.softmax(target_preds, dim=1).argmax(dim=1)
            accuracy = torch.eq(softmax_target_preds, y_target_set_task).data.cpu().float().mean()
            metrics_dict['accuracy'][-1].append(accuracy)
            metrics_dict['total_per_step_losses'][-1].append(target_loss)

            if not training_phase:
                self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses_dict=metrics_dict)

        return losses, np.array(metrics_dict['per_task_preds'])

    def trainable_parameters(self, exclude_params_with_string=None):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for name, param in self.named_parameters():
            if exclude_params_with_string is not None:
                if param.requires_grad and all(
                        list([exclude_string not in name for exclude_string in exclude_params_with_string])):
                    yield param
            else:
                if param.requires_grad:
                    yield param

    def trainable_names_parameters(self, exclude_params_with_string=None):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for name, param in self.named_parameters():
            if exclude_params_with_string is not None:
                if param.requires_grad and all(
                        list([exclude_string not in name for exclude_string in exclude_params_with_string])):
                    yield (name, param)
            else:
                if param.requires_grad:
                    yield (name, param)

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        if self.args.train_in_stages == True and epoch > self.args.train_learning_rates_as_a_stage_num_epochs:
            self.optimizer = optim.Adam(self.trainable_parameters(), lr=self.args.meta_learning_rate,
                                        weight_decay=self.args.weight_decay, amsgrad=False)

            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                  T_max=self.args.total_epochs,
                                                                  eta_min=self.args.min_learning_rate)

        losses, per_task_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                              use_second_order=self.args.second_order and
                                                               epoch > self.args.first_order_to_second_order_epoch,
                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                              num_steps=self.args.number_of_training_steps_per_iter,
                                              training_phase=True)
        return losses, per_task_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                              use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                              num_steps=self.args.number_of_evaluation_steps_per_iter,
                                              training_phase=False)

        return losses, per_task_preds

    def meta_update(self, meta_system_loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """

        self.optimizer.zero_grad()
        self.zero_grad()

        meta_system_loss.backward()

        if 'imagenet' in self.args.dataset_name:
            for name, param in self.trainable_names_parameters(self.exclude_list):

                if param.grad.data is None:
                    print('No gradients for parameter', name)

                if self.clip_grads and param.grad.data is not None and param.requires_grad:
                    param.grad.data.clamp_(-10, 10)

        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch
            # print(epoch, self.optimizer)

        if not self.training:
            self.train()

        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set, y_support_set_original, y_target_set_original = data_batch
        x_support_set_augmented = torch.Tensor(x_support_set_augmented).float().to(device=self.device)
        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        x_target_set_augmented = torch.Tensor(x_target_set_augmented).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)
        y_support_set_original = torch.Tensor(y_support_set_original).long().to(device=self.device)
        y_target_set_original = torch.Tensor(y_target_set_original).long().to(device=self.device)

        data_batch = (
            x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set,
            y_support_set_original,
            y_target_set_original)

        losses, per_task_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(meta_system_loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.zero_grad()

        return losses, per_task_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_support_set_augmented, x_target_set, x_target_set_augmented, y_support_set, y_target_set, y_support_set_original, y_target_set_original = data_batch

        x_support_set_augmented = torch.Tensor(x_support_set_augmented).float().to(device=self.device)
        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)
        y_support_set_original = torch.Tensor(y_support_set_original).long().to(device=self.device)
        y_target_set_original = torch.Tensor(y_target_set_original).long().to(device=self.device)

        data_batch = (
            x_support_set, x_support_set_augmented, x_target_set, y_support_set, y_target_set, y_support_set_original,
            y_target_set_original)

        losses, per_task_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        return losses, per_task_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath, map_location=self.args.device)
        net = dict(state['network'])

        state['network'] = OrderedDict(net)

        self.load_state_dict(state_dict=state['network'])
        self.starting_iter = state['current_iter']
        return state