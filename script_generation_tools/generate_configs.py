import os
import shutil
from collections import namedtuple

seed_list = [0, 1, 2]  # 3, 4]

config = namedtuple('config', 'num_classes_per_set '
                              ' exclude_param_string weight_decay num_target_set_steps '
                              'batch_size inner_loop_optimizer_type conditional_information '
                              'train_update_steps '
                              'val_update_steps total_epochs output_spatial_dimensionality use_channel_wise_attention '
                              'inner_loop_learning_rate experiment_name num_filters conv_padding load_into_memory '
                              'learnable_bn_gamma learnable_bn_beta num_stages num_blocks_per_stage '
                              'learnable_learning_rates learnable_betas '
                              'classifier_type num_samples_per_support_class num_samples_per_target_class '
                              'num_support_sets overwrite_classes_in_each_task class_change_interval ')


def string_generator(string_list):
    output_string = "["
    i = 0
    for string_entry in string_list:
        if i < len(string_list) - 1:
            output_string += "\"{}\",".format(string_entry)
        else:
            output_string += "\"{}\"".format(string_entry)

        i += 1

    output_string += "]"
    return output_string


experiment_conditional_information_config = [["preds"]]
# "task_embedding", "penultimate_layer_features"]
configs_list = []

for (n_way, k_shot, batch_size) in [(5, 1, 1)]:
    for use_channel_wise_attention in [True, False]:
        for classifier_type in ['maml++_low-end', 'vgg-fine-tune-scratch', 'vgg-fine-tune-pretrained',
                                'maml++_high-end', 'vgg-matching_network']:
            for (num_support_sets, class_change_interval, overwrite_classes_in_each_task) in \
                    [(1, 1, False), (3, 1, True), (3, 1, False), (5, 1, True),
                     (5, 1, False), (10, 1, True), (10, 1, False),
                     (3, 3, True), (5, 5, True), (10, 10, True),
                     (4, 2, False), (8, 2, False)]:
                if classifier_type == 'vgg-fine-tune-scratch':
                    total_epochs = 5
                else:
                    total_epochs = 250
                if not 'vgg-matching_network' in classifier_type:
                    for output_dim in [5, 20]:
                        configs_list.append(
                            config(num_classes_per_set=5,
                                   num_samples_per_support_class=k_shot,
                                   num_samples_per_target_class=5,
                                   batch_size=batch_size, train_update_steps=5, val_update_steps=0,
                                   inner_loop_learning_rate=0.01,
                                   conv_padding=1, num_filters=48, load_into_memory=True,
                                   conditional_information=[],
                                   num_target_set_steps=0, weight_decay=0.0001,
                                   total_epochs=total_epochs,
                                   exclude_param_string=string_generator(
                                       ["None"]),
                                   experiment_name='default_{}_way_{}_{}_shot_{}_{}_{}_{}_LSLR_conditioned'.format(
                                       n_way, k_shot, classifier_type, "_".join([], ),
                                       overwrite_classes_in_each_task,
                                       num_support_sets, class_change_interval), learnable_bn_beta=True,
                                   learnable_bn_gamma=True,
                                   num_stages=4, num_blocks_per_stage=0,
                                   inner_loop_optimizer_type='LSLR', learnable_betas=False,
                                   learnable_learning_rates=True,
                                   output_spatial_dimensionality=output_dim,
                                   use_channel_wise_attention=use_channel_wise_attention,
                                   classifier_type=classifier_type,
                                   num_support_sets=num_support_sets,
                                   overwrite_classes_in_each_task=overwrite_classes_in_each_task,
                                   class_change_interval=class_change_interval))

                        for conditional_information in experiment_conditional_information_config:
                            if classifier_type == 'maml++_high-end':
                                configs_list.append(
                                    config(num_classes_per_set=5,
                                           num_samples_per_support_class=k_shot,
                                           num_samples_per_target_class=5,
                                           batch_size=batch_size, train_update_steps=5, val_update_steps=1,
                                           inner_loop_learning_rate=0.01,
                                           conv_padding=1, num_filters=48, load_into_memory=True,
                                           conditional_information=string_generator(conditional_information),
                                           num_target_set_steps=1, weight_decay=0.0001,
                                           total_epochs=total_epochs,
                                           exclude_param_string=string_generator(
                                               ["None"]),
                                           experiment_name='SCA_{}_way_{}_{}_shot_{}_{}_{}_{}_LSLR_conditioned'.format(
                                               n_way, k_shot, classifier_type, "_".join(conditional_information),
                                               overwrite_classes_in_each_task, num_support_sets,
                                               class_change_interval),
                                           learnable_bn_beta=True,
                                           learnable_bn_gamma=True,
                                           num_stages=4, num_blocks_per_stage=0,
                                           inner_loop_optimizer_type='LSLR', learnable_betas=False,
                                           learnable_learning_rates=True,
                                           output_spatial_dimensionality=output_dim,
                                           use_channel_wise_attention=use_channel_wise_attention,
                                           classifier_type=classifier_type,
                                           num_support_sets=num_support_sets,
                                           overwrite_classes_in_each_task=overwrite_classes_in_each_task,
                                           class_change_interval=class_change_interval))

                else:
                    configs_list.append(
                        config(num_classes_per_set=5,
                               num_samples_per_support_class=k_shot,
                               num_samples_per_target_class=5,
                               batch_size=batch_size, train_update_steps=5, val_update_steps=0,
                               inner_loop_learning_rate=0.01,
                               conv_padding=1, num_filters=48, load_into_memory=True,
                               conditional_information=[],
                               num_target_set_steps=0, weight_decay=0.0001,
                               total_epochs=total_epochs,
                               exclude_param_string=string_generator(
                                   ["None"]),
                               experiment_name='default_{}_way_{}_{}_shot_{}_{}_{}_{}_LSLR_conditioned'.format(
                                   n_way, k_shot, classifier_type, "_".join([], ), overwrite_classes_in_each_task,
                                   num_support_sets, class_change_interval), learnable_bn_beta=True,
                               learnable_bn_gamma=True,
                               num_stages=4, num_blocks_per_stage=0,
                               inner_loop_optimizer_type='LSLR', learnable_betas=False,
                               learnable_learning_rates=True,
                               output_spatial_dimensionality=1,
                               use_channel_wise_attention=use_channel_wise_attention,
                               classifier_type=classifier_type,
                               num_support_sets=num_support_sets,
                               overwrite_classes_in_each_task=overwrite_classes_in_each_task,
                               class_change_interval=class_change_interval))

experiment_templates_json_dir = '../experiment_template_config/'
experiment_config_target_json_dir = '../experiment_config/'

if os.path.exists(experiment_config_target_json_dir):
    shutil.rmtree(experiment_config_target_json_dir)

os.makedirs(experiment_config_target_json_dir)

if not os.path.exists(experiment_config_target_json_dir):
    os.makedirs(experiment_config_target_json_dir)


def fill_template(script_text, config):
    for key, value in config.items():
        script_text = script_text.replace('${}$'.format(key), str(value).lower())

    return script_text


def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.read()

    return template


def write_text_to_file(text, filepath):
    with open(filepath, mode='w') as filewrite:
        filewrite.write(text)


for subdir, dir, files in os.walk(experiment_templates_json_dir):
    for template_file in files:
        for seed_idx in seed_list:
            filepath = os.path.join(subdir, template_file)

            for config in configs_list:
                loaded_template_file = load_template(filepath=filepath)
                config_dict = config._asdict()
                config_dict['train_seed'] = seed_idx
                config_dict['val_seed'] = seed_idx
                config_dict['experiment_name'] = "{}_{}_{}".format(template_file.replace(".json", ''),
                                                                   config.experiment_name, seed_idx)
                cluster_script_text = fill_template(script_text=loaded_template_file,
                                                    config=config_dict)

                cluster_script_name = '{}/{}_{}_{}.json'.format(experiment_config_target_json_dir,
                                                                template_file.replace(".json", ''),
                                                                config.experiment_name, seed_idx)
                print(cluster_script_name, seed_idx)
                cluster_script_name = os.path.abspath(cluster_script_name)
                write_text_to_file(cluster_script_text, filepath=cluster_script_name)
