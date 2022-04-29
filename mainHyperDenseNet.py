
from os.path import isfile, join
import os
import numpy as np
from sampling import reconstruct_volume
from sampling import my_reconstruct_volume
from sampling import load_data_trainG
from sampling import load_data_test

from utils import generate_temp

import torch
import torch.nn as nn
from HyperDenseNet import *
from medpy.metric.binary import dc,hd
import argparse

import pdb
from torch.autograd import Variable
from progressBar import printProgressBar
import nibabel as nib

import multiprocessing
from torch.utils.tensorboard import SummaryWriter



def evaluateSegmentation(gt,pred, eval_labels): # almost completely rewritten by ziyao
    pred = pred.astype(dtype='int')
    # numClasses = np.unique(gt)
    numClasses = len(eval_labels) # ziyao
    print("unique classes are: " + str(numClasses))

    dsc = np.zeros((1, numClasses - 1))

    # for i_n in range(1,len(numClasses)):

    ind = 1
    for i_n in eval_labels[1:]:  # ziyao
        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==i_n)]=1
        y_c[np.where(pred==i_n)]=1

        dsc[0, ind - 1] = dc(gt_c, y_c)
        ind += 1

    return dsc
    
def numpy_to_var(x):
    torch_tensor = torch.from_numpy(x).type(torch.FloatTensor)
    
    if torch.cuda.is_available():
        torch_tensor = torch_tensor.cuda()
    return Variable(torch_tensor)
    
def inference(network, moda_n, moda_g, imageNames, epoch, folder_save, number_modalities, seg_labels, toMerge, eval_labels):
    '''root_dir = './Data/MRBrainS/DataNii/'
    model_dir = 'model'

    moda_1 = root_dir + 'Training/T1'
    moda_2 = root_dir + 'Training/T1_IR'
    moda_3 = root_dir + 'Training/T2_FLAIR'
    moda_g = root_dir + 'Training/GT'''
    network.eval()
    softMax = nn.Softmax() # ziyao added dim
    numClasses = len(seg_labels) # ziyao changed this hardcode
    if torch.cuda.is_available():
        softMax.cuda()
        network.cuda()

    # def getInd(lab):  # created by ziyao
    #     return np.where(np.array(seg_labels) == lab)[0][0] if (lab in np.array(seg_labels)) else 0
    #
    # cvt = np.vectorize(getInd)
    # eval_labels = cvt(eval_labels) # convert the evaluation labels into ind(segmentation_labels)
    # print("the converted eval labels are: " + str(eval_labels))

    dscAll = np.zeros((len(imageNames), len(eval_labels) - 1))  # 1st class is the background!! ziyao
    for i_s in range(len(imageNames)):

        if number_modalities == 2:
            patch_1, patch_2, patch_g, img_shape = load_data_test(moda_n, moda_g, imageNames[i_s], number_modalities)  # hardcoded to read the first file. Loop this to get all files
        if number_modalities == 3:
            patch_1, patch_2, patch_3, patch_g, img_shape = load_data_test([moda_n], moda_g, imageNames[i_s], number_modalities) # hardcoded to read the first file. Loop this to get all files

        patchSize = 27
        patchSize_gt = 9

        x = np.zeros((0, number_modalities, patchSize, patchSize, patchSize))
        x = np.vstack((x, np.zeros((patch_1.shape[0], number_modalities, patchSize, patchSize, patchSize))))
        x[:, 0, :, :, :] = patch_1
        x[:, 1, :, :, :] = patch_2
        if (number_modalities==3):
            x[:, 2, :, :, :] = patch_3
        
        pred_numpy = np.zeros((0,numClasses,patchSize_gt,patchSize_gt,patchSize_gt))
        pred_numpy = np.vstack((pred_numpy, np.zeros((patch_1.shape[0], numClasses, patchSize_gt, patchSize_gt, patchSize_gt))))
        totalOp = len(imageNames)*patch_1.shape[0]
        pred = network(numpy_to_var(x[0,:,:,:,:]).view(1,number_modalities,patchSize,patchSize,patchSize))
        for i_p in range(patch_1.shape[0]):
            pred = network(numpy_to_var(x[i_p,:,:,:,:].reshape(1,number_modalities,patchSize,patchSize,patchSize)))
            pred_y = softMax(pred)
            pred_numpy[i_p,:,:,:,:] = pred_y.cpu().data.numpy()
            printProgressBar(i_s * ((totalOp + 0.0) / len(imageNames)) + i_p + 1, totalOp,
                             prefix="[Validation] ",
                             length=15)

        # To reconstruct the predicted volume
        extraction_step_value = 9
        pred_classes = np.argmax(pred_numpy, axis=1)

        pred_classes = pred_classes.reshape((len(pred_classes), patchSize_gt, patchSize_gt, patchSize_gt))
        #bin_seg = reconstruct_volume(pred_classes, (img_shape[1], img_shape[2], img_shape[3]))
        bin_seg = my_reconstruct_volume(pred_classes,
                                        (img_shape[1], img_shape[2], img_shape[3]),
                                        patch_shape=(27, 27, 27),
                                        extraction_step=(extraction_step_value, extraction_step_value, extraction_step_value))
        # bin_seg is the segmentation
        bin_seg = bin_seg[:,:,extraction_step_value:img_shape[3]-extraction_step_value]

        ###############################ziyao change segmentation content here#######################################
        # tuples = np.array([(21,2), (61,41), (170,16)])

        # for tuple in toMerge:  # merge labels
        #     bin_seg[bin_seg == getInd(tuple[0])] = getInd(tuple[1])


        ###############################ziyao change segmentation content here#######################################

        # load GT
        gt = nib.load(moda_g + '/' + imageNames[i_s]).get_data()
        ###############################ziyao change gt content here#######################################
        # print("previous gt shape:" + str(gt.shape))
        # print("previous gt type="+str(type(gt)))
        # generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18,
        #                               21, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
        #                               61, 62, 63])

        # def transformg(lab):
        #     return getInd(lab)
        #
        # trans = np.vectorize(transformg)
        # gt = cvt(gt)
        for set in toMerge:
            cvtTo = set[1]
            cvtArr = set[0]
            for label in cvtArr:
                gt[gt == label] = cvtTo
        print("converted gt labels" + str(np.unique(gt)))
        # print("after gt shape:" + str(gt.shape))
        # print("after gt type="+str(type(gt)))
        ###############################ziyao change gt content here#######################################

        img_pred = nib.Nifti1Image(bin_seg, np.eye(4))
        img_gt = nib.Nifti1Image(gt, np.eye(4))

        img_name = imageNames[i_s].split('.nii')
        name = 'Pred_' + img_name[0] + '_Epoch_' + str(epoch) + '.nii.gz'

        namegt = 'GT_' + img_name[0] + '_Epoch_' + str(epoch) + '.nii.gz'

        if not os.path.exists(folder_save + 'Segmentations/'):
            os.makedirs(folder_save + 'Segmentations/')

        if not os.path.exists(folder_save + 'GT/'):
            os.makedirs(folder_save + 'GT/')

        nib.save(img_pred, folder_save + 'Segmentations/' + name)
        nib.save(img_gt, folder_save + 'GT/' + namegt)

        dsc = evaluateSegmentation(gt, bin_seg, eval_labels)
        print("The dice scores are: " + str(dsc))
        dscAll[i_s, :] = dsc # problem here is that len(segmentation labels) > unique(GT_after_resetting) due to
        # 1mo GTs not having cerebellum seperations

    return dscAll
        
def runTraining(opts):
    print('' * 41)
    print('~' * 50)
    print('~~~~~~~~~~~~~~~~~  PARAMETERS ~~~~~~~~~~~~~~~~')
    print('~' * 50)
    print('  - Number of image modalities: {}'.format(opts.numModal))
    print('  - Number of classes: {}'.format(opts.numClasses))
    print('  - Segmentation labels: {}'.format(str(opts.segmentation_labels))) # ziyao added
    print('  - Merge tuples: {}'.format(str(opts.merge_tuples))) # ziyao added
    print('  - Evaluation labels: {}'.format(str(opts.eval_labels))) # ziyao added
    print('  - Merge labels: {}'.format(str(opts.merge_labels)))
    print('  - Directory to load images: {}'.format(opts.root_dir))
    for i in range(len(opts.modality_dirs)):
        print('  - Modality {}: {}'.format(i+1,opts.modality_dirs[i]))
    print('  - Directory to save results: {}'.format(opts.save_dir))
    print('  - To model will be saved as : {}'.format(opts.modelName))
    print('-' * 41)
    print('  - Number of epochs: {}'.format(opts.numEpochs)) # numEpochs, variable name changed by ziyao
    print('  - Batch size: {}'.format(opts.batchSize))
    print('  - Number of samples per epoch: {}'.format(opts.numSamplesEpoch))
    print('  - Learning rate: {}'.format(opts.l_rate))
    print('' * 41)

    print('-' * 41)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 41)
    print('' * 40)

    writer = SummaryWriter("runs/boardalt")

    samplesPerEpoch = opts.numSamplesEpoch
    batch_size = opts.batchSize

    lr = opts.l_rate
    epoch = opts.numEpochs
    
    root_dir = opts.root_dir
    model_name = opts.modelName

    if not (len(opts.modality_dirs)== opts.numModal): raise AssertionError

    moda_1 = root_dir + 'Training/' + opts.modality_dirs[0]
    moda_2 = root_dir + 'Training/' + opts.modality_dirs[1]

    if (opts.numModal == 3):
        moda_3 = root_dir + 'Training/' + opts.modality_dirs[2]

    moda_g = root_dir + 'Training/GT'

    print(' --- Getting image names.....')
    print(' - Training Set: -')
    if os.path.exists(moda_1):
        imageNames_train = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f))]
        imageNames_train.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_train)):
            print(' - {}'.format(imageNames_train[i])) 
    else:
        raise Exception(' - {} does not exist'.format(moda_1))

    moda_1_val = root_dir + 'Validation/' + opts.modality_dirs[0]
    moda_2_val = root_dir + 'Validation/' + opts.modality_dirs[1]

    if (opts.numModal == 3):
        moda_3_val = root_dir + 'Validation/' + opts.modality_dirs[2]
    moda_g_val = root_dir + 'Validation/GT'

    print(' --------------------')
    print(' - Validation Set: -')
    if os.path.exists(moda_1):
        imageNames_val = [f for f in os.listdir(moda_1_val) if isfile(join(moda_1_val, f))]
        imageNames_val.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_val)):
            print(' - {}'.format(imageNames_val[i])) 
    else:
        raise Exception(' - {} does not exist'.format(moda_1_val))
          
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = opts.numClasses
    
    # Define HyperDenseNet
    # To-Do. Get as input the config settings to create different networks
    if (opts.numModal == 2):
        hdNet = HyperDenseNet_2Mod(num_classes)
    if (opts.numModal == 3):
        hdNet = HyperDenseNet(num_classes)
    #

    '''try:
        hdNet = torch.load(os.path.join(model_name, "Best_" + model_name + ".pkl"))
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    softMax = nn.Softmax() # dim added by ziyao
    CE_loss = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        hdNet.cuda()
        softMax.cuda()
        CE_loss.cuda()

    # To-DO: Check that optimizer is the same (and same values) as the Theano implementation
    optimizer = torch.optim.Adam(hdNet.parameters(), lr=lr, betas=(0.9, 0.999))

    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    numBatches = int(samplesPerEpoch/batch_size)
    dscAll = []
    saveLoss = []
    addgraph = True

    for e_i in range(epoch):
        # generate images

        # p = multiprocessing.Process(target=generate_temp(2, merge_labels=opts.merge_labels))
        # p.start()
        # p.join()
        # p.terminate()
        generate_temp(opts.generate_num, merge_labels=opts.merge_labels)

        hdNet.train()
        
        lossEpoch = []

        if (opts.numModal == 2):
            imgPaths = [moda_1, moda_2]


        if (opts.numModal == 3):
            imgPaths = [moda_1, moda_2, moda_3]

        x_train, y_train, img_shape = load_data_trainG(imgPaths, moda_g, imageNames_train, samplesPerEpoch, opts.numModal) # hardcoded to read the first file. Loop this to get all files. Karthik

        ######################################ziyao modified labels to np arange here##############################
        # print("previous y type="+str(type(y_train)))
        # print("previous y shape=" + str(y_train.shape))
        # # generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18,
        # #                               21, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
        # #                               61, 62, 63])
        #
        # def transform(lab):
        #     segmentation_labels = np.array(opts.segmentation_labels)
        #     return np.where(segmentation_labels == lab)[0][0] if (lab in segmentation_labels) else 0
        #
        # trans = np.vectorize(transform)
        # y_train = trans(y_train)
        # #
        #
        # print("converted segs " + str(np.unique(y_train)))
        # print("after y type=" + str(type(y_train)))
        # print("after y shape=" + str(y_train.shape))
        #
        # #  the segmentations have the following problems:
        # #  (1) Segmentation classes must be from 0-numClasses (solved by converting to the index on segmentation_labels,
        # #  which is added as a param for merge_train)
        # #  (2) The segmentations contains labels that should be merged. This could be converted before evaluating
        # #  (partially solved).
        # #  (3) Some labels present in the GT are not in segmentation_labels (solved by resetting them to background 0)
        # #  (4) The evaluation assumes that len(segmentation labels)==uniaue(GT_after_resetting), which is often not the case.
        # #  A possible solution is to evaluate only the labels present in both the GT and segmentation_labels
        # #  regardless of whether it is present in the actual segmentations. (are there exceptions?)
        # #  A better solution may be to create an evaluation_labels param.
        # # Above all solved
        #
        ######################################ziyao modified labels to np arange here##############################


        # print(np.sum(x_train))
        # print("!!!!!!!!!!!!!!")

        # add patch example to tensorflow board
        MRI_exp = numpy_to_var(x_train[1 * batch_size:1 * batch_size + batch_size, :, :, :, :])
        print(MRI_exp[0,:,13,:,:].shape)
        writer.add_image("training patch", MRI_exp[0,1,13:17,:,:], dataformats='CHW')
        writer.close()

        if addgraph:
            writer.add_graph(hdNet, MRI_exp)
            writer.close()
            addgraph = False

        for b_i in range(numBatches):
            optimizer.zero_grad()
            hdNet.zero_grad()
            
            MRIs         = numpy_to_var(x_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:,:])

            Segmentation = numpy_to_var(y_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:])

            segmentation_prediction = hdNet(MRIs)
            # print("shape of posterior??="+str(segmentation_prediction.shape))
            # print("number of classes="+str(num_classes))
            # print("the previous prediction is: " + str(segmentation_prediction[0, 15, 6:8, 6:8, 6:8]))
            # print("the sum is: " + str(np.sum(segmentation_prediction.cpu().data.numpy())))
            predClass_y = softMax(segmentation_prediction)
            # print("the after prediction is: " + str(segmentation_prediction[0, 15, 6:8, 6:8, 6:8]))
            # print("the sum is: " + str(np.sum(segmentation_prediction.cpu().data.numpy())))
            # print("segmentation shape="+str(Segmentation.shape))
            # print(Segmentation)

            # To adapt CE to 3D
            # LOGITS:
            segmentation_prediction = segmentation_prediction.permute(0,2,3,4,1).contiguous()
            segmentation_prediction = segmentation_prediction.view(segmentation_prediction.numel() // num_classes, num_classes)
            
            CE_loss_batch = CE_loss(segmentation_prediction, Segmentation.view(-1).type(torch.cuda.LongTensor))


            loss = CE_loss_batch # log this (maybe) and visualize: Done
            loss.backward()
            
            optimizer.step()
            lossEpoch.append(CE_loss_batch.cpu().data.numpy())

            # print("!!!!!!!!!!!the batch loss is: ")
            # print(CE_loss_batch.cpu().data.numpy())
            # save batch loss
            batchloss = CE_loss_batch.cpu().data.numpy()
            saveLoss.append(batchloss)
            # writer.add_scalar('Batch loss', batchloss, e_i*numBatches+b_i) # TODO: too many points crashes the file transfer
            # writer.close()

            printProgressBar(b_i + 1, numBatches,
                             prefix="[Training] Epoch: {} ".format(e_i),
                             length=15)
              
            del MRIs
            del Segmentation
            del segmentation_prediction
            del predClass_y

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        np.save(os.path.join(model_name, model_name + '_loss.npy'), dscAll)

        print(' Epoch: {}, loss: {}'.format(e_i,np.mean(lossEpoch)))

        if (e_i%10)==0:

            if (opts.numModal == 2):
                moda_n = [moda_1_val, moda_2_val]
            if (opts.numModal == 3):
                moda_n = [moda_1_val, moda_2_val, moda_3_val]

            dsc = inference(hdNet,moda_n, moda_g_val, imageNames_val,e_i, opts.save_dir,opts.numModal, opts.segmentation_labels, opts.merge_tuples, opts.eval_labels)

            dscAll.append(dsc)

            print(' Metrics: DSC(mean): {} per class: 1({}) 2({}) 3({})'.format(np.mean(dsc),dsc[0][0],dsc[0][1],dsc[0][2]))
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            
            np.save(os.path.join(model_name, model_name + '_DSCs.npy'), dscAll)

        d1 = np.mean(dsc)
        if (d1>0.60):
            if not os.path.exists(model_name):
                os.makedirs(model_name)
                
            torch.save(hdNet, os.path.join(model_name, "Best2_" + model_name + ".pkl"))

        np.save(file='/home/ziyaos/SSG_HDN/HyperDenseNet_pytorch/batchlosses_at9.npy',
                arr=saveLoss)  # save the loss for each batch
        writer.add_scalar('epoch loss', np.mean(saveLoss[-(numBatches):]), e_i)
        writer.close()

        # if (100+e_i%20)==0: # what does this mean? This will never be triggered. Changed to the code below
        if (e_i % 20 == 0) and (e_i != 0):
            lr = lr/2
            print(' Learning rate decreased to : {}'.format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

                        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./Data/MRBrainS/DataNii/', help='directory containing the train and val folders')
    parser.add_argument('--modality_dirs', nargs='+', default=['T1','T2_FLAIR'], help='subdirectories containing the multiple modalities')
    parser.add_argument('--save_dir', type=str, default='./Results/', help='directory ot save results')
    parser.add_argument('--modelName', type=str, default='HyperDenseNet_2Mod', help='name of the model')
    parser.add_argument('--numModal', type=int, default=2, help='Number of image modalities')
    parser.add_argument('--numClasses', type=int, default=4, help='Number of classes (Including background)')
    parser.add_argument('--numSamplesEpoch', type=int, default=1000, help='Number of samples per epoch')
    parser.add_argument('--numEpochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=10, help='Batch size')
    parser.add_argument('--l_rate', type=float, default=0.0002, help='Learning rate')

    opts = parser.parse_args()
    print(opts)
    
    runTraining(opts)
