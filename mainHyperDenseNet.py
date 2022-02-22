



from os.path import isfile, join
import os
import numpy as np
from sampling import reconstruct_volume
from sampling import my_reconstruct_volume
from sampling import load_data_trainG
from sampling import load_data_test

import torch
import torch.nn as nn
from HyperDenseNet import *
from medpy.metric.binary import dc,hd
import argparse

import pdb
from torch.autograd import Variable
from progressBar import printProgressBar
import nibabel as nib

def evaluateSegmentation(gt,pred):
    pred = pred.astype(dtype='int')
    numClasses = np.unique(gt)

    dsc = np.zeros((1, len(numClasses) - 1))

    for i_n in range(1,len(numClasses)):
        gt_c = np.zeros(gt.shape)
        y_c = np.zeros(gt.shape)
        gt_c[np.where(gt==i_n)]=1
        y_c[np.where(pred==i_n)]=1

        dsc[0, i_n - 1] = dc(gt_c, y_c)
    return dsc
    
def numpy_to_var(x):
    torch_tensor = torch.from_numpy(x).type(torch.FloatTensor)
    
    if torch.cuda.is_available():
        torch_tensor = torch_tensor.cuda()
    return Variable(torch_tensor)
    
def inference(network, moda_n, moda_g, imageNames, epoch, folder_save, number_modalities, seg_labels):
    '''root_dir = './Data/MRBrainS/DataNii/'
    model_dir = 'model'

    moda_1 = root_dir + 'Training/T1'
    moda_2 = root_dir + 'Training/T1_IR'
    moda_3 = root_dir + 'Training/T2_FLAIR'
    moda_g = root_dir + 'Training/GT'''
    network.eval()
    softMax = nn.Softmax(dim=1) # ziyao added dim
    numClasses = len(seg_labels) # ziyao changed this hardcode
    if torch.cuda.is_available():
        softMax.cuda()
        network.cuda()

    dscAll = np.zeros((len(imageNames), numClasses - 1))  # 1 class is the background!!
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

        bin_seg = bin_seg[:,:,extraction_step_value:img_shape[3]-extraction_step_value]
        gt = nib.load(moda_g + '/' + imageNames[i_s]).get_data()
        ###############################ziyao change gt content here#######################################
        print("previous gt shape:" + str(gt.shape))
        print("previous gt type="+str(type(gt)))
        # generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18,
        #                               21, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
        #                               61, 62, 63])

        def transformg(lab):
            segmentation_labels = np.array(seg_labels)
            return np.where(segmentation_labels == lab)[0][0] if (lab in segmentation_labels) else 0

        trans = np.vectorize(transformg)
        gt = trans(gt)
        print("converted gt labels" + str(np.unique(gt)))
        print("after gt shape:" + str(gt.shape))
        print("after gt type="+str(type(gt)))
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

        dsc = evaluateSegmentation(gt,bin_seg)
        dscAll[i_s, :] = dsc

    return dscAll
        
def runTraining(opts):
    print('' * 41)
    print('~' * 50)
    print('~~~~~~~~~~~~~~~~~  PARAMETERS ~~~~~~~~~~~~~~~~')
    print('~' * 50)
    print('  - Number of image modalities: {}'.format(opts.numModal))
    print('  - Number of classes: {}'.format(opts.numClasses))
    print('  - Segmentation labels: {}'.format(str(opts.segmentation_labels))) # ziyao added
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

    softMax = nn.Softmax(dim=1) # dim added by ziyao
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
    for e_i in range(epoch):
        hdNet.train()
        
        lossEpoch = []

        if (opts.numModal == 2):
            imgPaths = [moda_1, moda_2]


        if (opts.numModal == 3):
            imgPaths = [moda_1, moda_2, moda_3]

        x_train, y_train, img_shape = load_data_trainG(imgPaths, moda_g, imageNames_train, samplesPerEpoch, opts.numModal) # hardcoded to read the first file. Loop this to get all files. Karthik

        ######################################ziyao modified labels to np arange here##############################
        print("previous y type="+str(type(y_train)))
        print("previous y shape=" + str(y_train.shape))
        # generation_labels = np.array([0, 14, 15, 16, 24, 77, 85, 170, 172, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18,
        #                               21, 26, 28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60,
        #                               61, 62, 63])

        def transform(lab):
            segmentation_labels = np.array(opts.segmentation_labels)
            return np.where(segmentation_labels == lab)[0][0] if (lab in segmentation_labels) else 0

        trans = np.vectorize(transform)
        y_train = trans(y_train)
        print("converted segs " + str(np.unique(y_train)))
        print("after y type=" + str(type(y_train)))
        print("after y shape=" + str(y_train.shape))
        ######################################ziyao modified labels to np arange here##############################


        print(np.sum(x_train))
        print("!!!!!!!!!!!!!!")
        for b_i in range(numBatches):
            optimizer.zero_grad()
            hdNet.zero_grad()
            
            MRIs         = numpy_to_var(x_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:,:])

            Segmentation = numpy_to_var(y_train[b_i*batch_size:b_i*batch_size+batch_size,:,:,:])

            segmentation_prediction = hdNet(MRIs)

            print("shape of posterior??="+str(segmentation_prediction.shape))
            print("number of classes="+str(num_classes))
            predClass_y = softMax(segmentation_prediction)
            print("segmentation shape="+str(Segmentation.shape))
            print(Segmentation)

            # To adapt CE to 3D
            # LOGITS:
            segmentation_prediction = segmentation_prediction.permute(0,2,3,4,1).contiguous()
            segmentation_prediction = segmentation_prediction.view(segmentation_prediction.numel() // num_classes, num_classes)
            
            CE_loss_batch = CE_loss(segmentation_prediction, Segmentation.view(-1).type(torch.cuda.LongTensor))
            
            loss = CE_loss_batch
            loss.backward()
            
            optimizer.step()
            lossEpoch.append(CE_loss_batch.cpu().data.numpy())

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

            dsc = inference(hdNet,moda_n, moda_g_val, imageNames_val,e_i, opts.save_dir,opts.numModal, opts.segmentation_labels)

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

        if (100+e_i%20)==0:
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
