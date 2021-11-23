
from __future__ import print_function

import torch
from pytorch_model_summary import summary

import models


def setup_background_models(device,image_height, image_width,complexity = False, test_model = True):
    """creates, tests and return background model before training  """

    torch.backends.cudnn.benchmark = True

    netBE = models.Background_Encoder(image_height, image_width,  complexity)
    netBG = models.Background_Generator(image_height, image_width, complexity)

    netBE.eval()
    netBG.eval()
    print(f'loading models on device {device}')
    netBE.to(device)
    netBG.to(device)
    if  test_model:
        test_image = torch.zeros((1, 3, image_height, image_width)).to(device)
        background_latents = netBE(test_image)
        background_test = netBG(background_latents)
        print(f'description background encoder')
        print(summary(netBE, test_image.to(device), show_input=False))
        print(f'description background generator')
        background_latents = netBE(test_image.to(device))
        print(summary(netBG, background_latents, show_input=False))

    return netBE, netBG

def load_background_checkpoint(netBE,netBG,optimizers,hyperparameters,
                               background_training_mode):
    """ loads and returns a saved pretrained model"""

    print('loading background checkpoint')
    saved_background_checkpoint_path = hyperparameters['saved_background_checkpoint_path']
    checkpoint = torch.load(saved_background_checkpoint_path)
    encoder_state_dict = checkpoint['encoder_state_dict']
    generator_state_dict = checkpoint['generator_state_dict']
    netBE.load_state_dict(encoder_state_dict)
    netBG.load_state_dict(generator_state_dict)
    if background_training_mode:
        optimizers.load_state_dict(checkpoint['optimizer_state_dict'])
    background_epoch = checkpoint['epoch']
    print('background checkpoint loaded')
    return background_epoch




