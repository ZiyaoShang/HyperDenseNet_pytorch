import os
import csv
import numpy as np
# from scipy.ndimage import label

# import keras.layers as KL
# from keras.models import Model
# import tensorflow as tf
# from ext.lab2im import utils
# import numpy.random as npr
# import nibabel as nib

# for i in range(50):
#     print(npr.uniform())


# all_labels = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53,
#               54, 58, 60, 61, 62, 63, 64]
# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_labels/randomness_test/predictions/10_perc_rand/merged_posterior/DCAN20_merged_posterior.nii.gz", im_only=True)[60, 60, 60, :])
# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_labels/randomness_test/predictions/10_perc_rand/T2priors/T2_image_posterior/DCAN20_T2w_brain_posteriors.nii.gz", im_only=True)[60, 60, 60, :])
# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_labels/randomness_test/predictions/10_perc_rand/T1priors/T1_image_posterior/DCAN20_T1w_brain_posteriors.nii.gz", im_only=True)[60, 60, 60, :])
# print(sorted([2,3,1,4,2,5,6,7,2,1]))
# print(utils.get_volume_info(path_volume=r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/1mo_Tmpl01_GT.nii.gz", return_volume=True)[0].shape)
# #
# #
# path_segs = utils.list_images_in_folder("/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final")
# for label_map in path_segs:
#     WM_map, aff, header = utils.load_volume(label_map, im_only=False)
#     utils.save_volume(volume=WM_map, aff=aff, header=header, path=label_map, dtype='float32')
#
# print(np.bincount(np.array([[1,2,40,9,9], [1,2,40,9,9]]).flatten()))
# arr = np.array([[1, 2, 2], [3, 4, 5]])
# arr2 = ([[1, 1, 3], [8, 0, 5]])
# print(np.where(arr > arr2, arr, arr2))
# loaded_model = tf.keras.models.load_model("/home/ziyaos/ziyao_data/new_labels/DCAN_SplitWM_model.h5")
# print(np.load(r"C:\Users\zs\SynthSeg\data\labels_classes_priors\SynthSeg_segmentation_labels.npy"))
# print(np.load(r"C:\Users\zs\SynthSeg\scripts\niral_scripts\generate_labels_midbrain.npy"))


# labels = [0, 14, 15, 16, 172, 2, 3, 4, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 47, 49, 50, 51, 52, 53, 54,
#           58, 60, 61]

# label_list, _ = utils.get_list_labels(label_list=labels, save_label_list=None,
#                                        FS_sort=True)
# a = np.array([2,3])
# b = np.array([[1,2,3,4,5], [1,2,3,4,5]])
# b[:, a] = [[0,9],[9,0]]
# print(b)
# print(label_list)
# print(np.random.uniform(low=2, high=1, size=5))
# print(np.array([[1]*3]))
# arr1 = np.load("/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/young/priors/prior_means.npy")
# print(arr1)
# print(arr1.shape)

import matplotlib.pyplot as plt
arr1 = np.load(r"C:\Users\zs\PycharmProjects\HyperDenseNet_pytorch\batchlosses_ori.npy")
arr1 = np.log(arr1)
print(len(arr1))
epo = []
ind = 0
tot = 0
for i in arr1:
    tot += i
    ind += 1
    if ind == 50*75:
        epo.append(tot/50/75)
        tot = 0
        ind = 0
plt.plot(epo)
plt.title("logged cross-entropy loss (epoch average): original learning rate")
plt.savefig(r"C:\Users\zs\PycharmProjects\HyperDenseNet_pytorch\original_epoch.jpg")
plt.show()
# print(arr1.shape)
# print(np.random.randint(1))
# print(np.array([1,2,3,4,5,6])[2:200])
# arr=np.array([[1,2,3],[1,2,3]])
# print(arr[0:100,:])
# arr2 = np.load("C:/Users/zs/ziyao_data/new_labels/WM_priors/T2_priors_WM_prev/prior_stds.npy")
# print(arr2 == arr1)

# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_labels/randomness_test/predictions/625_perc_rand/islands/DCAN02_islands.nii.gz", im_only=True)[70, 30:90, 30:90])
# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_labels/randomness_test/predictions/625_perc_rand/islands/DCAN12_islands.nii.gz", im_only=True)[70, 30:90, 30:90])
# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_labels/randomness_test/predictions/625_perc_rand/merged_posterior/DCAN02_merged_posterior.nii.gz", im_only=True)[70, 70, 75])
# arr = np.array([1, 2, 3, 4, 5])
# arr2 = np.array([0, 1, 1, 1, 2])
# print(arr[arr2])

# print(utils.load_volume(r"/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/gzipped_6mo/GT/pd_6moTemp09_Segmentation.nii.gz", im_only=True).shape)

# print(np.load(r"C:/Users/zs/ziyao_data/new_labels/WM_stem_priors/T1priors/prior_stds.npy"))

#
# vol06 = r"/home/ziyaos/ziyao_data/new_infant_data_2021/GT/6moTemp09_Segmentation.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for seg_2021,09: " + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/new_infant_data_2021/GT/6moTemp08_Segmentation.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for seg_2021,08: " + str(label_list) + " length: " + str(len(label_list)))
#
#
# vol06 = "/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/gzipped_6mo/T2"
# path_seg = utils.list_images_in_folder(vol06)
# for path in range(len(path_seg)):
#     label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=path_seg[path], FS_sort=True)
#     print("label of " + path_seg[path].split("/")[-1].split("_")[0] + "_" + path_seg[path].split("/")[-1].split("_")[1]  + " " + str(label_list) + " length: " + str(len(label_list)))

# img_nifti1 = nib.Nifti1Image.from_filename(r"/home/ziyaos/ziyao_data/new_infant_data_2021/GT/6moTemp09_Segmentation.nii.gz")
# utils.save_volume(volume=np.array(img_nifti1.dataobj), aff=img_nifti1.affine, header=img_nifti1.header, path=r"/home/ziyaos/ziyao_data/new_infant_data_2021/GT/temp.nii.gz", dtype='int')
# print(img_nifti1.get_fdata())
# label_list,_ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=r"/home/ziyaos/ziyao_data/new_infant_data_2021/GT/6moTemp09_Segmentation.nii.gz", FS_sort=True)
# print(np.sum((utils.load_volume(r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/8mo_Tmpl09_GT.nii.gz", im_only=True))==6))
# print(len(label_list))
# vol06 = r"/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/DCAN/convertedGT"
############################################################
# vol06 = r"/home/ziyaos/ziyao_data/test_EP/mask"
# path_seg = utils.list_images_in_folder(vol06)
# for path in range(len(path_seg)):
#     print(path_seg[path])
    # label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=path_seg[path], FS_sort=True)
    # print("label of " + path_seg[path].split("/")[-1].split("_")[0] + "_" + path_seg[path].split("/")[-1].split("_")[1] + " " + str(label_list) + " length: " + str(len(label_list)))
    # print("its size is: " + str(utils.load_volume(path_seg[path], im_only=True).shape))
    # print(np.sum(utils.load_volume(path_seg[path], im_only=True) == 62))
#############################################################
#
# vol06 = r"/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/gzipped_6mo/GT/pd_6moTemp07_Segmentation.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 6mo04, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))

# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/6mo_Tmpl03_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 6mo03, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/8mo_Tmpl04_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 8mo04, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/8mo_Tmpl05_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 8mo05, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/1mo_Tmpl01_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 1mo01, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/1mo_Tmpl02_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 1mo02, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/2mo_Tmpl01_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 2mo01, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# print(np.sum(utils.load_volume(r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/2mo_Tmpl01_GT.nii.gz", im_only=True) == 6))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/2mo_Tmpl02_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 2mo02, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))
#
# vol06 = r"/home/ziyaos/ziyao_data/wm_JLF_atlases/convertedGT/2mo_Tmpl03_GT.nii.gz"
# label_list, _ = utils.get_list_labels(label_list=None, save_label_list=None, labels_dir=vol06, FS_sort=True)
# print("the labels for 2mo03, wm_JLF_atlases" + str(label_list) + " length: " + str(len(label_list)))

# i = 0
# arr = np.array([0,1,2,3])
# print(np.where(arr==i)[0][0])

# def transformg(lab):
#     segmentation_labels = np.array([0, 14, 15, 16, 170, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42,
#                                     43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 61])
#     return np.where(segmentation_labels == lab)[0][0] if (lab in segmentation_labels) else 0
#
#
# trans = np.vectorize(transformg)
# gt = np.arange(100)
# gt = trans(gt)
# print("converted gt labels" + str(np.unique(gt)))
# print(100+40%20)