import os
import argparse

def get_gpu_config():
    with open(config_file, 'r') as read:
        lines = read.readlines()

    return int(lines[0].split(':')[1].replace(' ', ''))

def change_gpu_config(gpu_num):
    lines = ['gpu_flag: {}'.format(gpu_num)]

    with open(config_file, 'w') as write_file:
        write_file.writelines(lines)

config_file = os.path.join('/disk/scratch/antreas', 'gpu_config.txt')

parser = argparse.ArgumentParser(description='Welcome to the run N at a time script')
parser.add_argument('--gpu_flag', type=int)

args = parser.parse_args()

if not os.path.exists(config_file):
    lines = ['gpu_flag: -1']

    with open(config_file, 'w') as write_file:
        write_file.writelines(lines)


change_gpu_config(gpu_num=args.gpu_flag)

