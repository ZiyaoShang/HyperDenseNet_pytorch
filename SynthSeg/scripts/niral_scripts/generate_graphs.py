from ext.lab2im import utils
import os
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def intensity_statistics(path_labels, path_intensities):
    segs = utils.list_images_in_folder(path_labels)
    intensities = utils.list_images_in_folder(path_intensities)

    label_list, _ = utils.get_list_labels(label_list=None, labels_dir=segs[0], save_label_list=None,
                                          FS_sort=True)
    for label in label_list[1:]:
        fig, ax = plt.subplots(4, 5)
        print("label: " + str(label))
        for subject in range(len(segs)):
            # print(segs[subject])
            # print(intensities[subject])
            seg, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(segs[subject], return_volume=True)
            intensity, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(intensities[subject], return_volume=True)
            label_intensities = np.array(intensity[seg == label]).astype('int')
            x = int(subject/5)
            y = subject % 5

            ax[x, y].hist(label_intensities, bins=int((np.max(label_intensities)-np.min(label_intensities))/10))
            ax[x, y].set_title(str(subject))
            ax[x, y].set_yticks([])

        plt.tight_layout()
        fig.suptitle('T1 label ' + str(label), y=1.0)
        plt.savefig("/home/ziyaos/ziyao_data/new_labels/deliverables/temp/test_T1_label" + str(label) + ".png")
        plt.close(fig)


def intensity_statistics_fused(path_labels, path_intensities):
    segs = utils.list_images_in_folder(path_labels)
    intensities = utils.list_images_in_folder(path_intensities)

    mid_brain = [14, 15, 16, 172]
    left_brain = [2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28]
    right_brain = [41, 42, 43, 47, 49, 50, 51, 52, 53, 54, 58, 60]

    for label in mid_brain:
        fig, ax = plt.subplots(4, 5)
        print("label mid: " + str(label))
        for subject in range(len(segs)):
            # print(segs[subject])
            # print(intensities[subject])
            seg, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(segs[subject], return_volume=True)
            intensity, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(intensities[subject], return_volume=True)
            label_intensities = np.array(intensity[seg == label]).astype('int')

            x = int(subject/5)
            y = subject % 5
            ax[x, y].hist(label_intensities, bins=int((np.max(label_intensities)-np.min(label_intensities))/10))
            ax[x, y].set_title(str(subject))
            ax[x, y].set_yticks([])

        plt.tight_layout()
        fig.suptitle('T2 label ' + str(label), y=1.0)
        plt.savefig("/home/ziyaos/ziyao_data/new_labels/deliverables/temp/test_T2_label" + str(label) + ".png")
        plt.close(fig)

    for ind in range(len(left_brain)):
        fig, ax = plt.subplots(4, 5)
        label_left = left_brain[ind]
        label_right = right_brain[ind]
        print("label left: " + str(label_left) + " and " + "label right: " + str(label_right))

        for subject in range(len(segs)):
            # print(segs[subject])
            # print(intensities[subject])
            seg, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(segs[subject], return_volume=True)
            intensity, shape, aff, n_dims, n_channels, header, im_res = utils.get_volume_info(intensities[subject], return_volume=True)
            mask = np.logical_or(seg == label_left, seg == label_right)
            label_intensities = np.array(intensity[mask]).astype('int')

            x = int(subject/5)
            y = subject % 5
            ax[x, y].hist(label_intensities, bins=int((np.max(label_intensities)-np.min(label_intensities))/10))
            ax[x, y].set_title(str(subject))
            ax[x, y].set_yticks([])

        plt.tight_layout()
        fig.suptitle('T2 labels ' + str(label_left) + " and " + str(label_right), y=1.0)
        plt.savefig("/home/ziyaos/ziyao_data/new_labels/deliverables/temp/test_T2_label_" + str(label_left) + "_" + str(label_right) + ".png")
        plt.close(fig)


def saveboxplot(file, save, exclude, rows=None):
    dice = pd.read_csv(file).drop(columns=exclude)
    headers = np.array(dice.columns.values)[1:]
    # print(np.array(dice)[rows[0]:rows[1],:])
    data = np.array(dice)[rows[0]:rows[1], 1:]
    print("headers: " + str(headers))
    print("data: " + str(data))
    print("there are " + str(len(headers)) + " labels, and the shape of the data array is: " + str(data.shape))
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_xticklabels(headers)
    plt.ylim(ymax=1, ymin=0.5)
    ax.set_title(file.split('/')[-1].split('.')[0])

    plt.savefig(save)

def compareboxplot(file1, file2, labels, rows1, rows2, name1, name2, title, remove_col0=True, save=None):
    # the two files must have the exact same headings and the selected subjects must have the same number of rows.
    f1 = pd.read_csv(file1)
    f2 = pd.read_csv(file2)

    f1headers = np.array(f1.columns.values)[remove_col0:]
    f2headers = np.array(f2.columns.values)[remove_col0:]

    data1 = np.array(f1)[rows1[0]: rows1[1], remove_col0:]
    data2 = np.array(f2)[rows2[0]: rows2[1], remove_col0:]

    print("data1 has shape: " + str(data1.shape))
    print("data1 headers: " + str(f1headers) + " with length: " + str(len(f1headers)))

    print("data2 has shape: " + str(data2.shape))
    print("data2 headers: " + str(f2headers) + " with length: " + str(len(f2headers)))

    assert np.all(f1headers == f2headers), "different headers"

    # first dataset
    empty = np.zeros(data1.shape[0])

    for label in labels:
        assert np.any(f1headers == str(label)), "label " + str(label) + " does not exist"
        ind = np.argmax(f1headers == str(label))

        # 1d arrays for one specific label
        arr1 = data1[:, ind]
        empty = np.vstack((empty, arr1))

    full = np.transpose(empty[1:, :])
    print(full)
    print(full.shape)
    fig, ax = plt.subplots()
    box = ax.boxplot(full, positions=np.arange(len(labels))-0.15, patch_artist=True, widths=0.3)

    # second dataset
    ax1 = ax.twinx()
    empty1 = np.zeros(data1.shape[0])
    for label in labels:
        assert np.any(f2headers == str(label)), "label " + str(label) + " does not exist"
        ind = np.argmax(f2headers == str(label))

        # 1d arrays for one specific label
        arr2 = data2[:, ind]
        empty1 = np.vstack((empty1, arr2))
    full1 = np.transpose(empty1[1:, :])
    print(full1)
    box1 = ax1.boxplot(full1, positions=np.arange(len(labels))+0.15, patch_artist=True, widths=0.3)

    # other processing
    for patch in box['boxes']:
        patch.set_facecolor('red')

    for patch in box1['boxes']:
        patch.set_facecolor('green')

    ax.tick_params(axis='both', which='both', length=0)
    ax1.tick_params(axis='both', which='both', length=0)

    ax.set_ylim([0, 1])
    ax1.set_ylim([0, 1])

    ax.set_xticklabels(np.array(labels).astype('int'))
    ax.legend([box["boxes"][0], box1["boxes"][0]], [name1, name2], loc='lower right')
    fig.tight_layout()
    fig.suptitle(title, y=1.0)

    plt.savefig(save)




# intensity_statistics(path_labels="/home/ziyaos/ziyao_data/DCAN_padded/DCAN_roi",
#                      path_intensities="/home/ziyaos/ziyao_data/DCAN_padded/DCAN_T1")

# intensity_statistics_fused(path_labels="/home/ziyaos/ziyao_data/DCAN_padded/DCAN_roi",
#                            path_intensities="/home/ziyaos/ziyao_data/DCAN_padded/DCAN_T2")

# saveboxplot(
    # file=r"/home/ziyaos/ziyao_data/new_labels/randomness_test/evaluations/75_perc_rand/DCAN_padded_fuse/DCAN_padded_fuse_dice_75_rand_corrected_post_island.csv",
    # save=r"/home/ziyaos/ziyao_data/new_labels/deliverables/temp/75_perc_DCAN_padded_fuse_training_dice_post_island.png",
    # exclude=['0', '14', '15', '16', '172'],
    # rows=[10, -2])

# young: [14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41,	42,	43,	47,	49,	50,	51,	52,	53,	54, 58, 60]
# young rereordered: [2, 41, 3, 42, 8, 47, 4, 43, 28, 60, 10, 49, 11, 50, 12, 51, 13, 52, 26, 58,17, 53, 18, 54]

# old: [14,	15,	16,	2,	3,	4,	8,	10,	11,	12,	13,	17,	18,	26,	28,	41,	42,	43,	47,	49,	50,	51,	52,	53,	54,	58, 60, 7, 46],
# old rereordered: [2, 41, 3, 42, 7, 46, 8, 47, 4, 43, 28, 60, 10, 49, 11, 50, 12, 51, 13, 52, 26, 58,17, 53, 18, 54]

# r'/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/predict/stats/dice_older_merged_postisl.csv'
# '/home/ziyaos/ziyao_data/new_labels/rand_retraining/50_pred/jlf/pre_island_50_dice_retrain.csv'

compareboxplot(file1=r'/home/ziyaos/ziyao_data/new_labels/rand_retraining/50_pred/DCAN/pre_island_50_dice_retrain.csv',
               file2=r'/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/predict/stats/dice_1_2mo_merged_postisl.csv',
               labels=[2, 41, 3, 42, 8, 47, 4, 43, 28, 60, 10, 49, 11, 50, 12, 51, 13, 52, 26, 58, 17, 53, 18, 54],
               rows1=[10, 20],
               rows2=[5, 15],
               name1=r"one-month model",
               name2=r"mixed-age model",
               title="1-month model vs mixed-age model on 1-month data",
               save='/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/predict/stats/final_graphs/1_preisl_vs_mix_postisl_dice_DCAN.jpg')

