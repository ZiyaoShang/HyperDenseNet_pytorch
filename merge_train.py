
import os
import time
import numpy as np
from SynthSeg.ext.lab2im import utils
from SynthSeg.SynthSeg.brain_generator import BrainGenerator
from argparse import ArgumentParser
from SynthSeg.scripts.niral_scripts.argument_parser import get_argparser
import mainHyperDenseNet as hdn
from utils import generate_temp
import multiprocessing
import tensorflow as tf


# def generate_temp():
#     # number of images to generate
#     num_image = 75
#     # path of the input label map
#     labels_dir = "/home/ziyaos/SSG_HDN/training_labels"
#     # path where to save the generated image
#     result_label = "/home/ziyaos/SSG_HDN/root/Training/GT"
#     result_T1 = "/home/ziyaos/SSG_HDN/root/Training/T1s"
#     result_T2 = "/home/ziyaos/SSG_HDN/root/Training/T2s"
#
#     T1_means = "/home/ziyaos/SSG_HDN/T1merged/prior_means.npy"
#     T1_stds = "/home/ziyaos/SSG_HDN/T1merged/prior_stds.npy"
#
#     T2_means = "/home/ziyaos/SSG_HDN/T2merged/prior_means.npy"
#     T2_stds = "/home/ziyaos/SSG_HDN/T2merged/prior_stds.npy"
#
#     generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2,  3,   4,   5,   7,   8,  10,  11,  12,  13, 17, 18,
#                               21,  26,  28,  30,  31,  41,  42,  43,  44,  46,  47,  49,  50,  51,  52,  53,  54,  58,  60,
#                               61,  62,  63])
#     generation_classes = np.array([0, 1, 2, 3, 4, 5, 6, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 9, 22,
#                                         23, 24, 25, 9, 27, 28, 29, 13, 31, 32, 33, 34, 35, 36, 37, 38, 39, 9, 41, 42])
#
#     segmentation_labels = [0, 14, 15, 16, 170, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 46,
#                                 47, 49, 50, 51, 52, 53, 54, 58, 60, 61]
#
#
#     flipping = False  # whether to right/left flip the training label maps, this will take sided labels into account
#     # (so that left labels are indeed on the left when the label map is flipped)
#     scaling_bounds = .15  # the following are for linear spatial deformation, higher is more deformation
#     rotation_bounds = 15
#     shearing_bounds = .012
#     translation_bounds = False  # here we deactivate translation as we randomly crop the training examples
#     nonlin_std = 3.  # maximum strength of the elastic deformation, higher enables more deformation
#     nonlin_shape_factor = .04  # scale at which to elastically deform, higher is more local deformation
#
#     # bias field parameters
#     bias_field_std = .5  # maximum strength of the bias field, higher enables more corruption
#     bias_shape_factor = .025  # scale at which to sample the bias field, lower is more constant across the image
#
#     # acquisition resolution parameters
#     # the following parameters aim at mimicking data that would have been 1) acquired at low resolution (i.e. data_res),
#     # and 2) upsampled to high resolution in order to obtain segmentation at high res (see target_res).
#     # We do not such effects here, as this script shows training parameters to segment data at 1mm isotropic resolution
#     data_res = None
#     randomise_res = False
#     thickness = None
#     downsample = False
#     blur_range = 1.03  # we activate this parameter, which enables SynthSeg to be robust against small resolution variations
#
#     # no randomness when selecting the templetes for generation
#
#     T1_generator = BrainGenerator(labels_dir, generation_labels=generation_labels, prior_means=T1_means,
#                                   prior_stds=T1_stds, flipping=flipping, generation_classes=generation_classes,
#                                   scaling_bounds=scaling_bounds,
#                                   rotation_bounds=rotation_bounds,
#                                   shearing_bounds=shearing_bounds,
#                                   nonlin_std=nonlin_std,
#                                   nonlin_shape_factor=nonlin_shape_factor,
#                                   data_res=data_res,
#                                   thickness=thickness,
#                                   downsample=downsample,
#                                   blur_range=blur_range,
#                                   bias_field_std=bias_field_std,
#                                   bias_shape_factor=bias_shape_factor,
#                                   mix_prior_and_random=True,
#                                   prior_distributions='normal',
#                                   use_generation_classes=0.5)
#
#
#     for i in range(num_image):
#         start = time.time()
#         im, lab = T1_generator.generate_brain()
#         end = time.time()
#         print('generation {0:d} took {1:.01f}s'.format(i, end - start))
#         print(im.shape)
#         # save output image and label map
#         utils.save_volume(np.squeeze(im), T1_generator.aff, T1_generator.header,
#                           os.path.join(result_T1, 'brain_%s.nii' % i))
#         utils.save_volume(np.squeeze(lab), T1_generator.aff, T1_generator.header,
#                           os.path.join(result_label, 'brain_%s.nii' % i))
#
#         print("Saved Output.")
#
#     print("step two")
#     # sequential selection
#     T2_generator = BrainGenerator(result_label, generation_labels=generation_labels, prior_means=T2_means,
#                                   prior_stds=T2_stds, generation_classes=generation_classes,
#                                   data_res=data_res,
#                                   thickness=thickness,
#                                   downsample=downsample,
#                                   blur_range=blur_range,
#                                   bias_field_std=bias_field_std,
#                                   bias_shape_factor=bias_shape_factor,
#                                   mix_prior_and_random=True,
#                                   prior_distributions='normal',
#                                   use_generation_classes=0.5,
#                                   flipping=False,
#                                   apply_linear_trans=False,
#                                   scaling_bounds=0,
#                                   rotation_bounds=0,
#                                   shearing_bounds=0,
#                                   apply_nonlin_trans=False,
#                                   nonlin_std=0,
#                                   nonlin_shape_factor=0)
#
#     label_names = utils.list_images_in_folder(result_label)
#     for i in range(num_image):
#         im, lab = T2_generator.generate_brain()
#         print(im.shape)
#         # save output image and label map
#         brain_ind = label_names[i].split('/')[-1].split('.')[0].split('_')[-1]
#         print(brain_ind)
#         utils.save_volume(np.squeeze(im), T2_generator.aff, T2_generator.header,
#                           os.path.join(result_T2, "brain_" + brain_ind + ".nii"))
#
#         print("Saved Output.")
#
#     print("Generation finished, generated " + str(num_image) + " brains")


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# generate images
p = multiprocessing.Process(target=generate_temp(75))
p.start()
p.join()
p.terminate()

print("Now starting training...")
segmentation_labels = [0, 14, 15, 16, 170, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 21, 26, 28, 41, 42, 43, 46,
                       47, 49, 50, 51, 52, 53, 54, 58, 60, 61]
parser = ArgumentParser()

parser.add_argument('--save_dir', type=str, default='./Results/', help='directory ot save results')
# parser.add_argument('--modelName', type=str, default='HyperDenseNet_2Mod', help='name of the model')
#
# parser.add_argument('--root_dir', type=str, default='./Data/MRBrainS/DataNii/',
#                     help='directory containing the train and val folders')
# parser.add_argument('--modality_dirs', nargs='+', default=['T1', 'T2_FLAIR'],
#                     help='subdirectories containing the multiple modalities')
# parser.add_argument('--numModal', type=int, default=2, help='Number of image modalities')
# parser.add_argument('--numClasses', type=int, default=4, help='Number of classes (Including background)')
# parser.add_argument('--numSamplesEpoch', type=int, default=1000, help='Number of samples per epoch')
# parser.add_argument('--numEpochs', type=int, default=500, help='Number of epochs')
# parser.add_argument('--batchSize', type=int, default=10, help='Batch size')
# parser.add_argument('--l_rate', type=float, default=0.0002, help='Learning rate')

opts = parser.parse_args()
opts.modelName = "SSG_HDN_2Mod"
opts.root_dir = "/home/ziyaos/SSG_HDN/dup/root/"
opts.modality_dirs = ["T1s", "T2s"]
opts.numModal = 2
opts.numClasses = len(segmentation_labels)
opts.numSamplesEpoch = 75*50
opts.numEpochs = 50
opts.batchSize = 1
opts.l_rate = 0.002 # TODO: 10*
opts.save_dir = '/home/ziyaos/SSG_HDN/dup/res/'
opts.segmentation_labels = segmentation_labels
opts.merge_tuples = np.array([(21,2), (61,41), (170,16)])
opts.eval_labels = [0, 14, 15, 16, 172, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
# the evaluation labels must be a subset of the segmentation labels
print(opts)

# TODO: use Tensorflow board.

hdn.runTraining(opts)
