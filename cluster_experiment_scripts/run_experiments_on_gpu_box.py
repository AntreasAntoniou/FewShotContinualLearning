import argparse
import getpass
import json
import os
import subprocess
import time

from tabulate import tabulate
import GPUtil
import numpy as np
import tqdm
from colorama import Fore, Back, Style


os.environ['CUDA_VISIBLE_DEVICES'] = ""


parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--include', type=str)
parser.add_argument('--num_parallel_running_jobs', type=int)
parser.add_argument('--total_epochs_per_job', type=int)
parser.add_argument('--num_gpus_per_task', type=int)
args = parser.parse_args()



config_file = os.path.join('/disk/scratch/antreas/', 'gpu_config.txt')
kill_nvidia_processes = "python nvidia-smi-advanced.py | grep 's1473470' | awk '{ print $3 }' | xargs -n1 kill -TERM; fuser -k /dev/nvidia-uvm"

if not os.path.exists(config_file):
    lines = ['gpu_flag: -1']

    with open(config_file, 'w') as write_file:
        write_file.writelines(lines)

def get_gpu_config():
    with open(config_file, 'r') as read:
        lines = read.readlines()

    return int(lines[0].split(':')[1].replace(' ', ''))

gpu_config = get_gpu_config()


def extract_args_from_json(json_file_path):
    summary_filename = json_file_path.replace('\n', '')
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
    config_file_name = "{}/{}".format("../experiment_config", config_file_name)

    return config_file_name


def get_experiment_name_from_experiment_config(exp_config_name):
    args_dict = extract_args_from_json(json_file_path=exp_config_name)
    return args_dict['experiment_name']

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



def load_template(filepath):
    with open(filepath, mode='r') as filereader:
        template = filereader.readlines()

    return template


def check_if_experiment_with_name_is_running(script_name, currently_running_scripts):
    return script_name in currently_running_scripts


def get_metrics_from_exp_script(script_name, currently_running_scripts):
    exp_config_name = get_experiment_config_from_script_file(script_name=script_name)
    exp_name = get_experiment_name_from_experiment_config(exp_config_name=exp_config_name)
    epochs, stats = get_stats_from_experiment_name(experiment_name=exp_name)
    is_running = check_if_experiment_with_name_is_running(script_name, currently_running_scripts)
    is_running = "{}True{}".format(Fore.GREEN, Fore.WHITE) if is_running else "{}False{}".format(Fore.YELLOW, Fore.WHITE)
    return [epochs] + list(stats) + [is_running]


def print_progress_page(epoch_dict, currently_running_scripts):
    header = ['id', 'experiment_name', 'current_epoch', 'test_acc', 'test_loss', 'best_val_acc', 'best_val_loss', 'best_train_acc', 'best_train_loss', 'is running']

    epoch_dict = {key: get_metrics_from_exp_script(key, currently_running_scripts)[0] for key, value
                  in epoch_dict.items()}


    table_entries = [header]
    for idx, (key, value) in enumerate(epoch_dict.items()):
        outputs = list(get_metrics_from_exp_script(key, currently_running_scripts))
        outputs = ["{:.3f}".format(float(item)) for item in outputs[:-1]] + [outputs[-1]]
        table_entries.append([float(idx), key] + outputs)

    print(tabulate(table_entries, headers="firstrow", tablefmt="pretty", floatfmt=".2f"))

    # reset_position = ''.join((len(epoch_dict.keys()) + 1) * ['\033[F'])

    # print(reset_position)
    # print('\r')

    return epoch_dict


def get_total_epochs_completed(epoch_dict, currently_running_scripts):
    total_epochs = np.sum(
        [get_metrics_from_exp_script(key, currently_running_scripts)[0] for key, value in epoch_dict.items()])
    return total_epochs

def check_string(string_item, include_list):
    return all([item in string_item for item in include_list])

from copy import copy

student_id = getpass.getuser().encode()[:5]
include_list = args.include.split(" ")
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE, shell=True).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh") if check_string(str(item), include_list)]
list_of_all_scripts = copy(list_of_scripts)
epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0
total_epochs = args.total_epochs_per_job
num_jobs_in_queue = args.num_parallel_running_jobs
current_num_processed_scripts = 0
total_num_scripts = len(list_of_scripts)
list_of_running_processes = []
list_of_running_scripts = []
list_of_in_use_gpus = []



for item in list_of_scripts:
    print("to run", item)

epoch_dict = print_progress_page(epoch_dict, currently_running_scripts=list_of_running_scripts)

list_of_scripts = [key for key, value in epoch_dict.items() if
                   value < total_epochs and key not in list_of_running_scripts]

current_completed_epochs = 0
target_completed_epochs = len(list_of_scripts) * args.total_epochs_per_job
current_gpu_config = get_gpu_config()
num_jobs_in_queue = args.num_parallel_running_jobs if current_gpu_config == -1 else current_gpu_config

with tqdm.tqdm(total=target_completed_epochs, smoothing=0.0) as pbar_experiment:
    while True:
        while len(list_of_running_scripts) < num_jobs_in_queue and len(list_of_scripts) > 0:
            current_script = np.random.choice(list_of_scripts, size=(1))[0]
            gpu_to_use = GPUtil.getAvailable(order='first', limit=num_jobs_in_queue, maxLoad=0.01,
                                             maxMemory=0.01, includeNan=False,
                                             excludeID=[], excludeUUID=[])
            gpu_to_use = [i for i in gpu_to_use if i not in list_of_in_use_gpus]
            if len(gpu_to_use) > 0:
                # print("Using GPU with ID", gpu_to_use)
                script_to_run = np.random.choice(list_of_scripts, size=(1))[0]
                gpu_to_use = gpu_to_use[0:args.num_gpus_per_task]
                if len(gpu_to_use) > 1:
                    gpu_to_use_string = ','.join([str(gpu_idx) for gpu_idx in gpu_to_use])
                else:
                    gpu_to_use_string = str(gpu_to_use[0])

                out = subprocess.Popen(
                    args=["bash {script_name} {gpu_idx}".format(script_name=script_to_run,
                                                                gpu_idx=gpu_to_use_string)],
                    universal_newlines=False,
                    shell=True,
                    stdout=open("stdout_{}".format(script_to_run.replace('.sh', '')), mode="w+"),
                    stderr=open("stderr_{}".format(script_to_run.replace('.sh', '')), mode="w+"))

                list_of_in_use_gpus.extend(gpu_to_use)
                list_of_running_processes.append(out)
                list_of_running_scripts.append(script_to_run)
                print("running {} on GPU {}".format(script_to_run, gpu_to_use_string))
            del list_of_scripts[0]

        epoch_dict = print_progress_page(epoch_dict, currently_running_scripts=list_of_running_scripts)

        gpu_config = args.num_parallel_running_jobs if get_gpu_config() == -1 else get_gpu_config()
        # print(gpu_config, current_gpu_config, [item.pid for item in list_of_running_processes])
        if gpu_config != current_gpu_config:
            current_gpu_config = gpu_config
            try:
                if current_gpu_config < len(list_of_running_processes):
                    os.system(kill_nvidia_processes)
            except:
                pass

            if current_gpu_config == -1:
                num_jobs_in_queue = args.num_parallel_running_jobs
            else:
                num_jobs_in_queue = current_gpu_config

        idx_of_alive_processes = [i for i in range(len(list_of_running_processes)) if
                                  list_of_running_processes[i].poll() == None]

        list_of_running_processes = [list_of_running_processes[i] for i in idx_of_alive_processes]

        list_of_running_scripts = [list_of_running_scripts[i] for i in idx_of_alive_processes]

        list_of_in_use_gpus = [list_of_in_use_gpus[i] for i in idx_of_alive_processes]

        list_of_scripts = [key for key, value in epoch_dict.items() if
                           value < total_epochs and key not in list_of_running_scripts]

        # print(list_of_scripts, list_of_running_scripts)
        if current_completed_epochs != get_total_epochs_completed(epoch_dict=epoch_dict,
                                                                  currently_running_scripts=list_of_scripts):
            diff = get_total_epochs_completed(epoch_dict=epoch_dict,
                                              currently_running_scripts=list_of_scripts) - current_completed_epochs
            if diff < 0:
                print('Diff is ', diff)
            else:
                pbar_experiment.update(diff)
                current_num_processed_scripts = total_num_scripts - len(list_of_scripts)
                current_completed_epochs = get_total_epochs_completed(epoch_dict=epoch_dict,
                                                                      currently_running_scripts=list_of_scripts)

        time.sleep(5)
