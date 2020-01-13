import os
import subprocess
import time
import sys
from collections import defaultdict
import argparse
import json
import tqdm
import getpass
import numpy as np

parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--num_parallel_running_jobs', type=int)
parser.add_argument('--include', type=str)
parser.add_argument('--total_epochs_per_job', type=int)
args = parser.parse_args()


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
    config_file_name = execution_line.split("experiment_config/")[1].split(" ")[0]
    config_file_name = "{}/{}".format("../experiment_config", config_file_name).replace("\n", "")

    return config_file_name


def get_experiment_name_from_experiment_config(exp_config_name):
    args_dict = extract_args_from_json(json_file_path=exp_config_name)
    return args_dict['experiment_name']


def load_csv_get_dict(filepath):
    with open(filepath, "r+") as read_csv:
        lines = read_csv.readlines()
    try:
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
    except:
        data_dict = None
    return data_dict


def get_stats_from_experiment_name(experiment_name):
    total_epochs_done = 0
    test_acc = -1.
    test_loss = np.inf
    val_acc = 0.
    val_loss = np.inf
    train_acc = 0.
    train_loss = np.inf
    epochs = 0
    experiment_dir = "../{}".format(experiment_name)
    for subdir, dir, files in os.walk(experiment_dir):
        for file in files:
            if file.endswith("statistics.csv"):
                filepath = os.path.join(subdir, file)
                stats_dict = load_csv_get_dict(filepath)
                if stats_dict is not None:
                    train_acc = np.max(stats_dict['train_accuracy_mean'])
                    train_loss = np.min(stats_dict['train_loss_mean'])
                    val_acc = np.max(stats_dict['val_accuracy_mean'])
                    val_loss = np.min(stats_dict['val_loss_mean'])
                    epochs = len(stats_dict['train_loss_mean'])
                    total_epochs_done += epochs
            if file.endswith("test_summary.csv"):
                try:
                    filepath = os.path.join(subdir, file)
                    test_dict = load_csv_get_dict(filepath)
                    if test_dict is not None:
                        test_acc = np.max(test_dict['test_accuracy_mean'])
                        test_loss = np.min(test_dict['test_loss_mean'])
                except:
                    pass

    return epochs, (test_acc, test_loss, val_acc, val_loss, train_acc, train_loss)


def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.readlines()

    return template

def get_metrics_from_exp_script(script_name):
    exp_config_name = get_experiment_config_from_script_file(script_name=script_name)
    exp_name = get_experiment_name_from_experiment_config(exp_config_name=exp_config_name)
    epochs, stats = get_stats_from_experiment_name(experiment_name=exp_name)
    is_running = check_if_experiment_with_name_is_running(script_name)
    return epochs, stats, is_running

def print_progress_page(epoch_dict):
    header = "experiment_name, current_epoch, best_val_acc, best_val_loss, best_train_acc, best_train_loss"
    new_output_list = ["{}: {}".format(key, get_metrics_from_exp_script(key)) for key, value in epoch_dict.items()]


    for i in range(len(new_output_list) + 2):
        print(end="\033[F")

    for idx, item in enumerate(new_output_list):
        if idx == 0:
            print(header)
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(idx, item)


def get_total_epochs_completed(epoch_dict):
    total_epochs = np.sum([get_metrics_from_exp_script(key)[0] for key, value in epoch_dict.items()])
    return total_epochs


def check_string(string_item, include_list):
    return all([item in string_item for item in include_list])

student_id = getpass.getuser().encode()[:5]
include_list = args.include.split(" ")
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE, shell=True).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh") if check_string(str(item), include_list)]

epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0
total_epochs = args.total_epochs_per_job
num_jobs_in_queue = args.num_parallel_running_jobs

# print("\n".join(list_of_scripts))

while total_jobs_finished < total_epochs * len(list_of_scripts):
    curr_idx = 0
    with tqdm.tqdm(total=len(list_of_scripts)) as pbar_experiment:
        while curr_idx < len(list_of_scripts):
            number_of_jobs = 0
            result = subprocess.run(['squeue', '-l'], stdout=subprocess.PIPE)
            for line in result.stdout.split(b'\n'):
                if student_id in line:
                    number_of_jobs += 1

            if number_of_jobs < num_jobs_in_queue:
                while check_if_experiment_with_name_is_running(
                        experiment_name=list_of_scripts[curr_idx].decode("utf-8")) or epoch_dict[
                    list_of_scripts[curr_idx].decode("utf-8")] >= total_epochs:
                    curr_idx += 1
                    if curr_idx >= len(list_of_scripts):
                        curr_idx = 0

                str_to_run = 'sbatch {}'.format(list_of_scripts[curr_idx].decode("utf-8"))

                if list_of_scripts[curr_idx].decode("utf-8") in epoch_dict:
                    epoch_dict[list_of_scripts[curr_idx].decode("utf-8")], _, _ = get_metrics_from_exp_script(
                        list_of_scripts[curr_idx].decode("utf-8"))

                os.system(str_to_run)
                curr_idx += 1

            else:
                print_progress_page(epoch_dict)
