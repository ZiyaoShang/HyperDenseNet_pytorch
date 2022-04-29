import numpy as np
import nibabel as nib
import time
import os
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import pdb
import itertools
from SynthSeg.SynthSeg import brain_generator
from SynthSeg.ext.lab2im import utils

def generate_indexes(patch_shape, expected_shape) :
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i+1] * (expected_shape[i] // patch_shape[i+1]) for i in range(ndims-1)]

    idxs = [range(patch_shape[i+1], poss_shape[i] - patch_shape[i+1], patch_shape[i+1]) for i in range(ndims-1)]

    return itertools.product(*idxs)

    
def extract_patches(volume, patch_shape, extraction_step) :
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

# Double check that number of labels is continuous
def get_one_hot(targets, nb_classes):
    #return np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return np.swapaxes(np.eye(nb_classes)[np.array(targets)],0,3) # Jose. To have the same shape as pytorch (batch_size, numclasses,x,y,z)

def build_set(imageData) :
    num_classes = 9
    patch_shape = (27, 27, 27)
    extraction_step=(15, 15, 15)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    
    imageData_1 = np.squeeze(imageData[0,:,:,:])
    imageData_2 = np.squeeze(imageData[1,:,:,:])
    imageData_3 = np.squeeze(imageData[2,:,:,:])
    imageData_g = np.squeeze(imageData[3,:,:,:])

    num_classes = len(np.unique(imageData_g))
    x = np.zeros((0, 3, 27, 27, 27))
    #y = np.zeros((0, 9 * 9 * 9, num_classes)) # Karthik
    y = np.zeros((0, num_classes, 9, 9, 9)) # Jose
    
    #for idx in range(len(imageData)) :
    y_length = len(y)

    label_patches = extract_patches(imageData_g, patch_shape, extraction_step)
    label_patches = label_patches[label_selector]
    
    # Select only those who are important for processing
    valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)

    # Filtering extracted patches
    label_patches = label_patches[valid_idxs]

    x = np.vstack((x, np.zeros((len(label_patches), 3, 27, 27, 27))))
    #y = np.vstack((y, np.zeros((len(label_patches), 9 * 9 * 9, num_classes)))) # Karthik
    y = np.vstack((y, np.zeros((len(label_patches), num_classes, 9, 9, 9))))  # Jose
    
    for i in range(len(label_patches)) :
        #y[i+y_length, :, :] = get_one_hot(label_patches[i, : ,: ,:].astype('int'), num_classes)  # Karthik
        y[i, :, :, :, :] = get_one_hot(label_patches[i, : ,: ,:].astype('int'), num_classes)  # Jose
    del label_patches

    # Sampling strategy: reject samples which labels are only zeros
    T1_train = extract_patches(imageData_1, patch_shape, extraction_step)
    x[y_length:, 0, :, :, :] = T1_train[valid_idxs]
    del T1_train

    # Sampling strategy: reject samples which labels are only zeros
    T2_train = extract_patches(imageData_2, patch_shape, extraction_step)
    x[y_length:, 1, :, :, :] = T2_train[valid_idxs]
    del T2_train

    # Sampling strategy: reject samples which labels are only zeros
    Fl_train = extract_patches(imageData_3, patch_shape, extraction_step)
    x[y_length:, 2, :, :, :] = Fl_train[valid_idxs]
    del Fl_train

        
    return x, y

def reconstruct_volume(patches, expected_shape) :
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(generate_indexes(patch_shape, expected_shape)) :
        selection = [slice(coord[i], coord[i] + patch_shape[i+1]) for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img


def load_data_train(path1, path2, path3, pathg, imgName):
    
    X_train = []
    Y_train = []
    for num in range(len(imgName)):
        # Karthik
        #imageData_1 = nib.load(path1 + '/' + imgName[num]).get_data()
        #imageData_2 = nib.load(path2 + '/' + imgName[num]).get_data()
        #imageData_3 = nib.load(path3 + '/' + imgName[num]).get_data()
        #imageData_g = nib.load(pathg + '/' + imgName[num]).get_data()
        
        # Jose
        imageData_1 = nib.load(path1 + '/' + imgName).get_data()
        imageData_2 = nib.load(path2 + '/' + imgName).get_data()
        imageData_3 = nib.load(path3 + '/' + imgName).get_data()
        imageData_g = nib.load(pathg + '/' + imgName).get_data()
        num_classes = len(np.unique(imageData_g))

        imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
        img_shape = imageData.shape

        x_train, y_train = build_set(imageData)

        X_train.append(x_train)
        Y_train.append(y_train)

        del x_train
        del y_train


    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)

    X = np.concatenate(X_train, axis=0)
    del X_train
    Y = np.concatenate(Y_train, axis=0)
    del Y_train
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    #pdb.set_trace()
    return X[idx], Y[idx], img_shape


def load_data_test(path1, path2, path3, pathg, imgName):

	imageData_1 = nib.load(path1 + '/' + imgName).get_data()
	imageData_2 = nib.load(path2 + '/' + imgName).get_data()
	imageData_3 = nib.load(path3 + '/' + imgName).get_data()
	imageData_g = nib.load(pathg + '/' + imgName).get_data()

	num_classes = len(np.unique(imageData_g))
	
	imageData = np.stack((imageData_1, imageData_2, imageData_3, imageData_g))
	img_shape = imageData.shape

	patch_1 = extract_patches(imageData_1, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
	patch_2 = extract_patches(imageData_2, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
	patch_3 = extract_patches(imageData_3, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
	patch_g = extract_patches(imageData_g, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))

	x_train, y_train = build_set(imageData)

	return patch_1, patch_2, patch_3, patch_g, img_shape


def generate_temp(num, merge_labels=None):
    # number of images to generate
    num_image = num
    # path of the input label map
    labels_dir = "/home/ziyaos/SSG_HDN/training_labels"
    # path where to save the generated image
    result_label = "/home/ziyaos/SSG_HDN/root/Training/GT"
    result_T1 = "/home/ziyaos/SSG_HDN/root/Training/T1s"
    result_T2 = "/home/ziyaos/SSG_HDN/root/Training/T2s"

    T1_means = "/home/ziyaos/SSG_HDN/T1merged/prior_means.npy"
    T1_stds = "/home/ziyaos/SSG_HDN/T1merged/prior_stds.npy"

    T2_means = "/home/ziyaos/SSG_HDN/T2merged/prior_means.npy"
    T2_stds = "/home/ziyaos/SSG_HDN/T2merged/prior_stds.npy"
    #
    generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2,  3,   4,   5,   7,   8,  10,  11,  12,  13, 17, 18,
                              21,  26,  28,  30,  31,  41,  42,  43,  44,  46,  47,  49,  50,  51,  52,  53,  54,  58,  60,
                              61,  62,  63])
    generation_classes = np.array([0, 1, 2, 3, 4, 5, 6, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 9, 22,
                                        23, 24, 25, 9, 27, 28, 29, 13, 31, 32, 33, 34, 35, 36, 37, 38, 39, 9, 41, 42])

    segmentation_labels = [0, 14, 15, 16, 170, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 46,
                                47, 49, 50, 51, 52, 53, 54, 58, 60, 61]


    flipping = False  # whether to right/left flip the training label maps, this will take sided labels into account
    # (so that left labels are indeed on the left when the label map is flipped)
    scaling_bounds = .15  # the following are for linear spatial deformation, higher is more deformation
    rotation_bounds = 15
    shearing_bounds = .012
    translation_bounds = False  # here we deactivate translation as we randomly crop the training examples
    nonlin_std = 3.  # maximum strength of the elastic deformation, higher enables more deformation
    nonlin_shape_factor = .04  # scale at which to elastically deform, higher is more local deformation

    # bias field parameters
    bias_field_std = .5  # maximum strength of the bias field, higher enables more corruption
    bias_shape_factor = .025  # scale at which to sample the bias field, lower is more constant across the image

    # acquisition resolution parameters
    # the following parameters aim at mimicking data that would have been 1) acquired at low resolution (i.e. data_res),
    # and 2) upsampled to high resolution in order to obtain segmentation at high res (see target_res).
    # We do not such effects here, as this script shows training parameters to segment data at 1mm isotropic resolution
    data_res = None
    randomise_res = False
    thickness = None
    downsample = False
    blur_range = 1.03  # we activate this parameter, which enables SynthSeg to be robust against small resolution variations

    # no randomness when selecting the templetes for generation

    T1_generator = brain_generator.BrainGenerator(labels_dir, generation_labels=generation_labels, prior_means=None,
                                  prior_stds=None, flipping=flipping, generation_classes=generation_classes,
                                  scaling_bounds=scaling_bounds,
                                  rotation_bounds=rotation_bounds,
                                  shearing_bounds=shearing_bounds,
                                  nonlin_std=nonlin_std,
                                  nonlin_shape_factor=nonlin_shape_factor,
                                  data_res=data_res,
                                  thickness=thickness,
                                  downsample=downsample,
                                  blur_range=blur_range,
                                  bias_field_std=bias_field_std,
                                  bias_shape_factor=bias_shape_factor,
                                  mix_prior_and_random=True,
                                  prior_distributions='uniform',
                                  use_generation_classes=0.5)


    for i in range(num_image):
        start = time.time()
        im, lab = T1_generator.generate_brain()
        end = time.time()
        print('generation {0:d} took {1:.01f}s'.format(i, end - start))
        print(im.shape)
        # save output image and label map
        utils.save_volume(np.squeeze(im), T1_generator.aff, T1_generator.header,
                          os.path.join(result_T1, 'brain_%s.nii' % i))
        utils.save_volume(np.squeeze(lab), T1_generator.aff, T1_generator.header,
                          os.path.join(result_label, 'brain_%s.nii.gz' % i))

        print("Saved Output.")
    del T1_generator

    print("step two")
    # sequential selection
    T2_generator = brain_generator.BrainGenerator(result_label, generation_labels=generation_labels, prior_means=None,
                                  prior_stds=None, generation_classes=generation_classes,
                                  data_res=data_res,
                                  thickness=thickness,
                                  downsample=downsample,
                                  blur_range=blur_range,
                                  bias_field_std=bias_field_std,
                                  bias_shape_factor=bias_shape_factor,
                                  mix_prior_and_random=True,
                                  prior_distributions='uniform',
                                  use_generation_classes=0.5,
                                  flipping=False,
                                  apply_linear_trans=False,
                                  scaling_bounds=0,
                                  rotation_bounds=0,
                                  shearing_bounds=0,
                                  apply_nonlin_trans=False,
                                  nonlin_std=0,
                                  nonlin_shape_factor=0)

    label_names = utils.list_images_in_folder(result_label)
    for i in range(num_image):
        im, lab = T2_generator.generate_brain()
        print(im.shape)
        # save output image and label map
        brain_ind = label_names[i].split('/')[-1].split('.')[0].split('_')[-1]
        print(brain_ind)
        utils.save_volume(np.squeeze(im), T2_generator.aff, T2_generator.header,
                          os.path.join(result_T2, "brain_" + brain_ind + ".nii"))

        print("Saved Output.")
    del T2_generator

    print("Generation finished, generated " + str(num_image) + " brains")

    # convert into large labels
    if merge_labels is not None:
        print("converting into " + str(len(merge_labels)) + " labels ")
        label_names = utils.list_images_in_folder(result_label)
        print(label_names)
        for lbmap in label_names:
            print(str(lbmap))
            volume, aff, header = utils.load_volume(path_volume=lbmap, im_only=False)
            for set in merge_labels:
                print(str(set))
                cvtTo = set[1]
                cvtArr = set[0]
                for label in cvtArr:
                    volume[volume == label] = cvtTo
            assert np.array_equal(np.unique(volume), np.arange(len(merge_labels))), str(np.unique(volume))
            print("saving...")
            utils.save_volume(np.squeeze(volume), aff, header, lbmap)
        print("convert finished")
