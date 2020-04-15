import argparse
import getpass
import json
import os
import subprocess
import time

import GPUtil
import numpy as np
import tqdm

parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--include', type=str, help='Include only scripts with names that contain the given strings, e.g. --include Resnet101 weight_decay layer_norm, will only select experiment scripts that include the words Reset101 AND weight_decay AND layer_norm')
parser.add_argument('--num_parallel_running_jobs', type=int)
parser.add_argument('--total_runs_per_experiment', type=int)
args = parser.parse_args()

def check_if_experiment_with_name_is_running(script_name, currently_running_scripts):
    return script_name in currently_running_scripts

def get_metrics_from_exp_script(script_name, currently_running_scripts):
    is_running = check_if_experiment_with_name_is_running(script_name, currently_running_scripts)
    return epoch_dict[script_name], is_running

def print_progress_page(epoch_dict, currently_running_scripts):
    header = "experiment_name: \t is_running"
    new_output_list = ["{}: \t {}".format(key, get_metrics_from_exp_script(key, currently_running_scripts)) for key, value
                       in epoch_dict.items()]

    for i in range(len(new_output_list) + 2):
        print(end="\033[F")

    for idx, item in enumerate(new_output_list):
        if idx == 0:
            print(header)
            print(
                "------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print(idx, item)

def check_string(string_item, include_list):
    return all([item in string_item for item in include_list])

student_id = getpass.getuser().encode()[:5]
include_list = args.include.split(" ")
list_of_scripts = [item for item in
                   subprocess.run(['ls'], stdout=subprocess.PIPE, shell=True).stdout.split(b'\n') if
                   item.decode("utf-8").endswith(".sh") if check_string(str(item), include_list)]

epoch_dict = {key.decode("utf-8"): 0 for key in list_of_scripts}
total_jobs_finished = 0
total_runs = args.total_runs_per_experiment
num_jobs_in_queue = args.num_parallel_running_jobs
current_num_processed_scripts = 0
total_num_scripts = len(list_of_scripts)
list_of_running_processes = []
list_of_running_scripts = []
list_of_in_use_gpus = []
for item in list_of_scripts:
    print("to run", item)

print_progress_page(epoch_dict, currently_running_scripts=list_of_running_scripts)


with tqdm.tqdm(total=len(list_of_scripts)) as pbar_experiment:
    while len(list_of_scripts) > 0 or len(list_of_running_scripts) > 0:
        while len(list_of_running_scripts) < num_jobs_in_queue and len(list_of_scripts) > 0:
            current_script = list_of_scripts[0].decode("utf-8")
            current_epochs = get_metrics_from_exp_script(current_script, currently_running_scripts=list_of_running_scripts)[0]
            if current_epochs < total_runs:

                gpu_to_use = []

                gpu_to_use = GPUtil.getAvailable(order='first', limit=num_jobs_in_queue, maxLoad=0.1,
                                                 maxMemory=0.1, includeNan=False,
                                                 excludeID=[], excludeUUID=[])
                gpu_to_use = [i for i in gpu_to_use if i not in list_of_in_use_gpus]
                if len(gpu_to_use) > 0:
                    #print("Using GPU with ID", gpu_to_use)
                    script_to_run = list_of_scripts[0].decode("utf-8")
                    gpu_to_use = gpu_to_use[0]
                    out = subprocess.Popen(
                        args=["bash {script_name} {gpu_idx}".format(script_name=script_to_run, gpu_idx=gpu_to_use)],
                        universal_newlines=False,
                        shell=True,
                        stdout=open("stdout_{}".format(time.time()), mode="w+"), stderr=open("stderr_{}".format(time.time()), mode="w+"))

                    list_of_in_use_gpus.append(gpu_to_use)
                    list_of_running_processes.append(out)
                    list_of_running_scripts.append(script_to_run)
                    print("running {} on GPU {}".format(script_to_run, gpu_to_use))
                    epoch_dict[current_script] += 1
            del list_of_scripts[0]

        print_progress_page(epoch_dict, currently_running_scripts=list_of_running_scripts)

        idx_of_alive_processes = [i for i in range(len(list_of_running_processes)) if list_of_running_processes[i].poll() == None]

        list_of_running_processes = [list_of_running_processes[i] for i in idx_of_alive_processes]

        list_of_running_scripts = [list_of_running_scripts[i] for i in idx_of_alive_processes]

        list_of_in_use_gpus = [list_of_in_use_gpus[i] for i in idx_of_alive_processes]

        if current_num_processed_scripts != total_num_scripts - len(list_of_scripts):
            diff = total_num_scripts - len(list_of_scripts) - current_num_processed_scripts
            pbar_experiment.update(diff)
            current_num_processed_scripts = total_num_scripts - len(list_of_scripts)

        time.sleep(30)
