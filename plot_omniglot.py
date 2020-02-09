import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import copy
import argparse
plt.style.use('ggplot')
parser = argparse.ArgumentParser(description="Harry Plotter")
parser.add_argument("--overwrite", type=str, default="false", help="use or not overwrite: true/false")
parser.add_argument("--cci", default=1, type=int, help="The CCI value: []")
parser.add_argument("--use_legend", default=False, action='store_true')
args = parser.parse_args()

#plt.figure(figsize = (16,9))
#plt.figure(figsize = (16,16))
# sns.set(style=args.style)

#Model,Overwrite,NSS,CCI,Acc,Std,Latex
file_path = 'omniglot_variant.csv'

if(args.overwrite=="true" or args.overwrite=="True"):
    x = [1, 3, 5, 10] #Number of Support Sets Per Task (NSS)
    with_overwrite=True
else:
    x = [1, 3, 5, 10] #Number of Support Sets Per Task (NSS)
    with_overwrite=False

names_dict = {'intrinsic-densenet-embedding-based': 'SCA',
               'standard-densenet-embedding-based': 'MAML++ H',
               'standard-vgg-based': 'MAML++ L',
               'standard-vgg-fine-tune-pretrained': 'Init + Tune',
               'standard-vgg-fine-tune-scratch': 'Pretrain + Tune',
               'standard-vgg-matching-network': 'ProtoNet'}

marker_dict = {'intrinsic-densenet-embedding-based': 'X',
               'standard-densenet-embedding-based': 'P',
               'standard-vgg-based': '.',
               'standard-vgg-fine-tune-pretrained': 'v',
               'standard-vgg-fine-tune-scratch': '^',
               'standard-vgg-matching-network': 's'}
                 
accuracy_dict = {'intrinsic-densenet-embedding-based':[],
                 'standard-densenet-embedding-based':[],
                 'standard-vgg-based':[],
                 'standard-vgg-fine-tune-pretrained':[],
                 'standard-vgg-fine-tune-scratch':[],
                 'standard-vgg-matching-network':[]}
std_dict = copy.deepcopy(accuracy_dict)

#Class-Change Interval (CCI)
with open(file_path) as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=',')
    for row in csv_reader:
        model=row['Model']
        accuracy=float(row['Acc'])
        std=float(row['Std'])
        overwrite=int(row['Overwrite'])
        NSS=int(row['NSS'])
        CCI=int(row['CCI'])
        if(overwrite==with_overwrite and CCI==args.cci): 
            print(model + " (NSS=" + str(NSS) + "): " + str(accuracy) + " +- " + str(std))
            accuracy_dict[model].append(accuracy)
            std_dict[model].append(std*2.0)
        elif(overwrite==False and CCI==1 and NSS==1): #FSL case
            accuracy_dict[model].append(accuracy)
            std_dict[model].append(std*2.0)

line_styles = ['-','--','-.',':']
for idx, (key, _) in enumerate(accuracy_dict.items()):

     print(accuracy_dict[key])

     plt.errorbar(x=x, y=accuracy_dict[key], yerr=std_dict[key],
                  marker=marker_dict[key], label=names_dict[key], alpha=0.95, linestyle=line_styles[idx%4], linewidth=3)


#plt.legend(['First', 'Second’], loc=4)
if args.use_legend:
    plt.legend(handlelength=5)

plt.ylim(bottom=0.0, top=105.0)
#plt.xlim(ymax = 250, ymin = 25)
plt.title("Omniglot (CCI=" + str(args.cci) + "; Overwrite=" + str(with_overwrite) + ")") # for title
plt.xlabel("NSS (#)", fontsize=14) # label for x-axis
plt.ylabel("Accuracy (%)", fontsize=14) # label for y-axis
# plt.autoscale(enable=True, axis='y', tight=None)
plt.gcf().set_size_inches(4, 4)
# plt.gcf().subplots_adjust(bottom=0.16)
plt.tight_layout()
plt.savefig('./figure_omniglot_' + str(args.cci) + '_' + str(with_overwrite) + '.png')




