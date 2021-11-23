from __future__ import print_function

import os
import torch
import torch.utils.data
import time
import shutil
from tqdm import tqdm
import cv2
import numpy as np
from argparse import ArgumentParser

import train
import utils
import dataset

# recommended options for speed optimization
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

def create_parser():
    parser = ArgumentParser()
    parser.add_argument('--input_path',
                        default=os.getcwd(),
                        help="Path to the input frames sequence directory")
    parser.add_argument('--results_dir_path',
                        default=os.getcwd(),
                        help="Path of the directory where to save the results")
    parser.add_argument('--beta', type=float, default=6, help="hyperparameter beta : bootstrap coefficient")
    parser.add_argument('--r', type=int, default=75, help="hyperparameter r")
    parser.add_argument('--tau_0', type=float,default = 0.24, help="hyperparameter tau_0")
    parser.add_argument('--tau_1', type=float, default=0.25, help="hyperparameter tau_1")
    parser.add_argument('--alpha_1', type=float, default=96/255, help="hyperparameter alpha_1")
    parser.add_argument('--alpha_2', type=float, default=7.0, help="hyperparameter alpha_2")
    parser.add_argument('--n_eval', type=int, default=2000, help="nb of iterations for evaluation")
    parser.add_argument('--n_simple', type=int, default=2500, help="nb of iterations for simple backgrounds")
    parser.add_argument('--n_complex', type=int, default=24000, help="minimum nb of iterations for complex backgrounds")
    parser.add_argument('--e_complex', type=int, default=20, help="minimum number of epochs for complex backgrounds")
    parser.add_argument('--supervised_mode', dest='unsupervised_mode', action='store_false')
    parser.add_argument('--n_iterations', type =int,default = 2500, help='number of iterations for supervised mode')
    parser.add_argument('--background_complexity', default=False,  help='background complexity for supervised mode')
    parser.set_defaults(unsupervised_mode=True)
    parser.add_argument('--use_trained_model',dest='train_model', action='store_false')
    parser.set_defaults(train_model=True)

    return parser

def compute_background_and_mask_using_trained_model(args,dataset,netBE, netBG, data,device):
    """ compute  backgrounds and masks from a batch of images using trained model
    data should be a batch of sample range 0-255"""

    images = {}
    real_images = data.to(device).type(torch.cuda.FloatTensor)  # range 0-255 Nx3xHxW RGB
    backgrounds_with_error_predictions = netBG(netBE(real_images))  # range 0-255 shape Nx4xHxW RGB+error prédiction
    backgrounds = backgrounds_with_error_predictions[:, 0:3, :, :]
    error_predictions = backgrounds_with_error_predictions[:, 3, :, :]  # NHW  0-255 float

    batch_size = data.shape[0]

    diffs = (torch.abs((real_images - backgrounds))).permute(0, 2, 3, 1).cpu().detach().numpy()  # NHW3 RGB
    l1_errors = (0.333 * np.sum(diffs, axis=3)).astype('uint8')  # NHW range 0-255
    images['l_1'] = l1_errors

    error_predictions = error_predictions.cpu().detach().numpy().astype('uint8')  # NHW 0-255
    images['noise'] = error_predictions

    NWC_backgrounds = backgrounds.permute(0, 2, 3, 1).cpu().detach().numpy().astype('uint8')  # NHW3 RGB
    backgrounds_opencv_format = NWC_backgrounds[:, :, :, ::-1]  # NHWC BGR
    images['backgrounds'] = backgrounds_opencv_format

    # placeholders for masks
    masks_before_post_processing = np.zeros((batch_size, dataset.image_height, dataset.image_width))
    masks = np.zeros((batch_size, dataset.image_height, dataset.image_width))

    for i in range(batch_size):

        corrected_dif = (np.maximum(0, l1_errors[i].astype('int16') - args.alpha_2 * error_predictions[i].astype('int16')))

        background_illumination = (torch.sum(backgrounds[i]) / (3 * dataset.image_height * dataset.image_width)).cpu().numpy() # range 0-255

        mask_before_post_processing = 255 * np.greater(3*corrected_dif, args.alpha_1 * background_illumination).astype('uint8')

        close_kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask_before_post_processing, cv2.MORPH_CLOSE, close_kernel)
        open_kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

        masks_before_post_processing[i, :, :] = mask_before_post_processing
        masks[i, :, :] = mask

    images['thresholded'] = masks_before_post_processing
    images['masks'] = masks

    return images

def compute_dynamic_backgrounds_and_masks(args,video_paths):
    """ train background model on train dataset if required and compute dynamic backgrounds on a test dataset """
    start_time = time.time()
    assert os.path.exists(video_paths['train_dataset']), 'wrong path for train dataset'
    assert os.path.exists(video_paths['test_dataset']), 'wrong path for test dataset'

    if args.train_model :
        if os.path.exists(video_paths['models']):
            shutil.rmtree(video_paths['models'])
        os.mkdir(video_paths['models'])

    for key in ['backgrounds', 'masks']:
        if os.path.exists(video_paths[key]):
            shutil.rmtree(video_paths[key])
        os.mkdir(video_paths[key])

    batch_size = 32
    device = torch.device("cuda", 0)
    model_path = os.path.join(video_paths['models'],'trained_model.pth')

    # if training required, train the model, otherwise load trained model
    if args.train_model:

        print(f"initialization of train dataset {video_paths['train_dataset']}")
        train_dataset = dataset.Image_dataset(video_paths['train_dataset'])

        netBE, netBG = train.train_dynamic_background_model(args, train_dataset,model_path,batch_size)

    print(f"initialization of test dataset {video_paths['test_dataset']}")
    test_dataset = dataset.Image_dataset(video_paths['test_dataset'])

    if not args.train_model:

        print(f'loading saved models from {model_path}')
        checkpoint = torch.load(model_path)
        encoder_state_dict = checkpoint['encoder_state_dict']
        generator_state_dict = checkpoint['generator_state_dict']
        complexity = checkpoint['complexity']
        netBE, netBG = utils.setup_background_models(device, test_dataset.image_height, test_dataset.image_width, complexity)
        netBE.load_state_dict(encoder_state_dict)
        netBG.load_state_dict(generator_state_dict)
        print('models succesfully loaded')

    netBE.eval()
    netBG.eval()

    with torch.no_grad():

        dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                 shuffle=False, num_workers=4,
                                                 drop_last=False, pin_memory=True)

        print(f"generating background and masks for {video_paths['test_dataset']}...")
        for i, test_images in enumerate(tqdm(dataloader)):
                images = compute_background_and_mask_using_trained_model(args,test_dataset,netBE, netBG, test_images, device)
                for j in range(test_images.shape[0]):
                    index = 1+i*batch_size+j
                    cv2.imwrite('%s/background_%06d.jpg' % (video_paths['backgrounds'], index), images['backgrounds'][j])
                    cv2.imwrite('%s/bin%06d.png' % (video_paths['masks'], index), images['masks'][j])

        print(f'total computation time ( training+mask generation ) : {time.time()- start_time} seconds')

if __name__ == "__main__":

        parser = create_parser()
        args = parser.parse_args()

        video_paths = {}

        # will train and generate backgrounds and masks on the full dataset
        video_paths['train_dataset'] = args.input_path
        video_paths['test_dataset'] = args.input_path

        results_path = args.results_dir_path
        video_paths['masks'] = os.path.join(results_path, 'results')
        video_paths['backgrounds'] = os.path.join(results_path, 'backgrounds')
        video_paths['models'] = os.path.join(results_path, 'models')

        compute_dynamic_backgrounds_and_masks(args, video_paths)
        print(f"foreground masks and predicted backgrounds are stored in directory {video_paths['masks']} and {video_paths['backgrounds']}")

