import argparse
import getpass
import json
import os
import subprocess
import time

import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--num_parallel_jobs', type=int)
parser.add_argument('--total_epochs', type=int)
args = parser.parse_args()
experiment_config_target_json_dir = 'experiment_config_files'


def check_if_experiment_with_name_is_running(experiment_name):
    result = subprocess.run(['squeue --name {}'.format(experiment_name), '-l'], stdout=subprocess.PIPE, shell=True)
    lines = result.stdout.split(b'\n')
    if len(lines) > 2:
        return True
    else:
        return False


def extract_args_from_json(json_file_path):
    summary_filename = json_file_path
    args_dict = dict()

    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict


def get_experiment_config_from_script_file(script_name):
    script_file = load_template(script_name)
    execution_line = script_file[-1]
    config_file_name = execution_line.split(experiment_config_target_json_dir)[1].split(" ")[0].replace("/", "")
    config_file_name = "../{}/{}".format(experiment_config_target_json_dir, config_file_name.replace("\n", ""))

    return config_file_name


def get_experiment_name_from_experiment_config(exp_config_name):
    args_dict = extract_args_from_json(json_file_path=exp_config_name)
    return args_dict['experiment_name']


def load_csv_get_dict(filepath):
    with open(filepath, "r+") as read_csv:
        lines = read_csv.readlines()
    if len(lines) > 1:
        keys = lines[0].split(",")
        keys = [item.replace("\n", "") for item in keys]
        data_dict = dict()
        for key in keys:
            data_dict[key] = []

        for line in lines[1:]:
            elements = line.split(",")
            for key, element in zip(keys, elements):
                try:
                    if "epoch_run_time" in key:
                        element = element.replace("\n", "")
                        data_dict[key].append(float(element))
                    else:
                        data_dict[key].append(float(element))
                except:
                    pass

        return data_dict
    else:
        return None


def get_stats_from_experiment_name(experiment_name):
    total_epochs_done = 0
    val_acc = 0.
    val_loss = np.inf
    train_acc = 0.
    train_loss = np.inf
    experiment_dir = "../{}".format(experiment_name)
    for subdir, dir, files in os.walk(experiment_dir):
        for file in files:
            if file == "summary.csv":
                filepath = os.path.join(subdir, file)
                stats_dict = load_csv_get_dict(filepath)
                if stats_dict is not None:
                    val_acc = np.max(stats_dict['val_acc'])
                    val_loss = np.min(stats_dict['val_loss'])
                    train_acc = np.max(stats_dict['train_acc'])
                    train_loss = np.min(stats_dict['train_loss'])
                    epochs = np.max(stats_dict['curr_epoch'])
                    total_epochs_done += epochs

    return total_epochs_done, val_acc, val_loss, train_acc, train_loss


def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.readlines()

    return template


def get_metrics_from_exp_script(script_name):
    exp_config_name = get_experiment_config_from_script_file(script_name=script_name)
    exp_name = get_experiment_name_from_experiment_config(exp_config_name=exp_config_name)
    stats = get_stats_from_experiment_name(experiment_name=exp_name)
    is_running = check_if_experiment_with_name_is_running(script_name)
    return stats, is_running


def print_progress_page(epoch_dict):
    print("")
    epoch_dict['experiment_name'] = ['current_epoch', 'best_val_acc', 'best_val_loss', 'best_train_acc',
                                     'best_train_loss']
    new_output_list = []
    max_length = [0 for i in range(5 + 1)]
    for exp_idx, (key, value) in enumerate(epoch_dict.items()):
        if 'experiment_name' not in key:
            metrics, is_running = get_metrics_from_exp_script(key)
        else:
            metrics = list(epoch_dict[key])
            is_running = "is_running"
        for idx, metric in enumerate([key] + list(metrics)):
            metric_length = len(str(metric))
            if metric_length > max_length[idx]:
                max_length[idx] = metric_length

    for exp_idx, (key, value) in enumerate(epoch_dict.items()):
        current_entry_string = []
        if 'experiment_name' not in key:
            metrics, is_running = get_metrics_from_exp_script(key)
        else:
            metrics = list(epoch_dict[key])
            is_running = "is_running"

        for idx, metric in enumerate([key] + list(metrics)):
            metric_length = len(str(metric))
            diff_length = max_length[idx] - metric_length
            space_padding = [" "] * diff_length
            metric = str(metric) + "".join(space_padding) + "  "
            current_entry_string.append(metric)
        current_entry_string = "".join(current_entry_string)
        new_output_list.append(current_entry_string)

    for i in range(len(new_output_list) + 1):
        print(end="\033[F")

    for item in new_output_list[::-1]:
        print(item)


def get_total_epochs_completed(epoch_dict):
    total_epochs = np.sum([get_metrics_from_exp_script(key)[0] for key, value in epoch_dict.items()])
    return total_epochs


student_id = getpass.getuser().encode()[:5]
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh")]

for script in list_of_scripts:
    print('sbatch', script.decode("utf-8"))

epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0

while total_jobs_finished < args.total_epochs * len(list_of_scripts):
    curr_idx = 0
    with tqdm.tqdm(total=len(list_of_scripts)) as pbar_experiment:
        while curr_idx < len(list_of_scripts):
            number_of_jobs = 0
            result = subprocess.run(['squeue', '-l'], stdout=subprocess.PIPE)
            for line in result.stdout.split(b'\n'):
                if student_id in line:
                    number_of_jobs += 1

            if number_of_jobs < args.num_parallel_jobs:
                while check_if_experiment_with_name_is_running(
                        experiment_name=list_of_scripts[curr_idx].decode("utf-8")) or epoch_dict[
                    list_of_scripts[curr_idx].decode("utf-8")] >= args.total_epochs:

                    curr_idx += 1
                    if curr_idx >= len(list_of_scripts):
                        curr_idx = 0

                str_to_run = 'sbatch {}'.format(list_of_scripts[curr_idx].decode("utf-8"))

                if list_of_scripts[curr_idx].decode("utf-8") in epoch_dict:
                    experiment_outputs, _ = get_metrics_from_exp_script(
                        list_of_scripts[curr_idx].decode("utf-8"))
                    epoch_dict[list_of_scripts[curr_idx].decode("utf-8")] = experiment_outputs[0]

                print_progress_page(epoch_dict)
                time.sleep(1)
                os.system(str_to_run)
                print(end="\033[F")
                curr_idx += 1

            else:
                print_progress_page(epoch_dict)
                time.sleep(1)
