# Very simple script showing how to generate new images with lab2im

import os
import time
import numpy as np
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator
from scripts.niral_scripts.argument_parser import get_argparser

# path of the input label map
path_label_map = "/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/training_labels"
# path where to save the generated image
result_dir = "/home/ziyaos/ziyao_data/new_labels/junk"

# generate an image from the label map.
# Because the image is spatially deformed, we also output the corresponding deformed label map.


generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2,  3,   4,   5,   7,   8,  10,  11,  12,  13, 17, 18,
                          21,  26,  28,  30,  31,  41,  42,  43,  44,  46,  47,  49,  50,  51,  52,  53,  54,  58,  60,
                          61,  62,  63])

# set path to segmentation labels (i.e. the ROI to segment and to compute the loss on)


# path_generation_classes = np.array([0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 6, 17, 18, 6, 20, 21, 22, 23, 24,
#                             25, 26, 27, 28, 29, 30, 6])

# path_generation_labels = [0, 7, 46]
# path_segmentation_labels = [0, 7, 46]
generation_classes = np.array([0, 1, 2, 3, 4, 5, 6, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 9, 22,
                                    23, 24, 25, 9, 27, 28, 29, 13, 31, 32, 33, 34, 35, 36, 37, 38, 39, 9, 41, 42])

print(generation_labels)
prior_means = "/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/T2merged/prior_means.npy"
prior_stds = "/home/ziyaos/ziyao_data/new_infant_data_2021/bootstraping/final/final_new_masking/merging_priors/T2/T2merged/prior_stds.npy"

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

brain_generator = BrainGenerator(path_label_map, generation_labels=generation_labels, prior_means=prior_means,
                                 prior_stds=prior_stds, flipping=flipping, generation_classes=generation_classes,
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
                                 prior_distributions='normal',
                                 use_generation_classes=0.5)
for n in range(40):
    start = time.time()
    im, lab = brain_generator.generate_brain()
    end = time.time()
    print('generation {0:d} took {1:.01f}s'.format(n, end - start))
    print(im.shape)
    # save output image and label map
    utils.save_volume(np.squeeze(im), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'brain_%s.nii.gz' % n))
    utils.save_volume(np.squeeze(lab), brain_generator.aff, brain_generator.header,
                      os.path.join(result_dir, 'labels_%s.nii.gz' % n))
    print("Saved Output.")
print("Done!!!")
