from __future__ import print_function

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import time
import utils
import torch.utils.data


def background_loss(args,real_images,backgrounds_with_error_prediction):
    """ background loss used during training
        input images format : tensor shape N,3,H,W range 0-255"""

    bs, nc, h, w = real_images.size()

    backgrounds = backgrounds_with_error_prediction[:, 0:3, :, :]
    pixel_errors = torch.sum(torch.nn.functional.smooth_l1_loss(real_images, backgrounds,  reduction='none', beta=3.0), dim=1)*(1/255.0) # range 0-3

    error_prediction = backgrounds_with_error_prediction[:, 3, :, :]*(1/255) # range 0-1
    error_prediction_error =  torch.nn.functional.smooth_l1_loss(error_prediction, pixel_errors.detach()*(1/3),  reduction='none', beta=3.0/255)

    with torch.no_grad(): #  weights computation do not require gradient

        soft_masks = torch.tanh(pixel_errors*(1/args.tau_1)) # range 0-1
        weight_logit = -args.beta*torch.nn.functional.avg_pool2d(soft_masks, 2 * (w // args.r) + 1,
                                                         stride=1, padding= w // args.r, count_include_pad=False) # range 0-1 BSxHxW
        normalized_pixel_weights = torch.exp(weight_logit)*(1/(h*w*bs))

    loss = torch.sum((pixel_errors+error_prediction_error) * normalized_pixel_weights)
    return loss


def evaluate_background_complexity_using_trained_model(args,dataset, netBE, netBG, batch_size):
    """evaluates whether the background changes are simple or complex using partially trained model"""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=4,
                                             drop_last=True, pin_memory=True,
                                             shuffle=True, persistent_workers=True)
    dataloader_iterator = iter(dataloader)
    if len(dataset) > batch_size * 15:
        number_of_batchs = 15
    else: # if number of frames <= 480, limit to one epoch
        number_of_batchs = len(dataset) // batch_size

    # placeholder for reconstructed backgrounds
    backgrounds_big_batch = torch.zeros(batch_size * number_of_batchs, 3, dataset.image_height, dataset.image_width)

    netBE.eval()

    with torch.no_grad():
        for i in range(number_of_batchs):
            real_images = next(dataloader_iterator).type(torch.cuda.FloatTensor)
            backgrounds_with_error_predictions = netBG(netBE(real_images))
            backgrounds_big_batch[i * batch_size:(i + 1) * batch_size, :, :, :] = backgrounds_with_error_predictions[:, 0:3,
                                                                              :, :]

    median_background = torch.median(backgrounds_big_batch, dim=0, keepdim=True)[0].expand_as(backgrounds_big_batch)

    pixel_errors = (1 / 255) * torch.sum(
        torch.nn.functional.l1_loss(backgrounds_big_batch, median_background, reduction='none'), dim=1)

    soft_masks = torch.tanh(pixel_errors*(1/args.tau_1))
    average_mask_background_error = torch.mean(soft_masks)

    netBE.train()

    if average_mask_background_error > args.tau_0:
        complex_background = True
    else:
        complex_background = False

    return complex_background


def background_training_loop(args,netBE, netBG, optimizer,
                             dataset, model_path,
                             batch_size, device,
                             number_of_steps, evaluation_step):

    traindataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                  num_workers=4,
                                                  drop_last=True, pin_memory=True,
                                                  shuffle=True, persistent_workers=True)

    print(f'starting autoencoder training loop')


    netBE.train()
    netBG.train()

    saved_network = False
    save_network = False
    complex_background = False

    learning_rate_reduction_step = (4 * number_of_steps) // 5
    learning_rate_is_reduced = False

    def lmbda(epoch):
        return 0.1
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    last_message_time = time.time()

    step = 0
    epoch = 0


    while True:

        for j, images in enumerate(traindataloader, 0):

            images = images.to(device)
            optimizer.zero_grad()
            backgrounds_with_error_prediction = netBG(netBE(images))  # range 0-255
            loss = background_loss(args,images, backgrounds_with_error_prediction)
            loss.backward()
            optimizer.step()

            if step > number_of_steps and saved_network == False:
                save_network = True

            if saved_network == True:
                print('training finished')
                return netBE, netBG, complex_background

            if time.time() - last_message_time > 15:
                last_message_time = time.time()
                print('[dataset %s][epoch %d][step %d/%d] loss: %.6f '
                      % (dataset.dir, epoch, step, number_of_steps,
                         loss))
            step += 1

            if step == evaluation_step and args.unsupervised_mode:

                complex_background = evaluate_background_complexity_using_trained_model(args,dataset, netBE, netBG, batch_size)

                if complex_background:
                    print('complex background detected, aborting current training and starting new training with updated model ')
                    return netBE, netBG, complex_background
                else:
                    print('simple background, finishing training')


            if save_network == True:
                torch.save({'complexity': netBE.complexity, 'encoder_state_dict': netBE.state_dict(),
                            'generator_state_dict': netBG.state_dict()
                            }, model_path)
                print('final model saved')
                saved_network = True

            if step >= learning_rate_reduction_step and learning_rate_is_reduced == False:
                scheduler.step()
                print(f'learning rate is now reduced (step {step})')
                learning_rate_is_reduced = True

        epoch = epoch + 1


def train_dynamic_background_model(args, dataset,model_path,batch_size):
    """ training function for dynamic background"""

    if args.unsupervised_mode:
        number_of_steps = args.n_simple
        evaluation_step = args.n_eval
        complexity = False
    else:
        number_of_steps = args.n_iterations
        complexity = args.background_complexity
        evaluation_step = 1e10 # no evaluation in supervised mode

    lr = 5e-4

    device = torch.device("cuda", 0)

    netBE, netBG = utils.setup_background_models(device, dataset.image_height,dataset.image_width,complexity)

    optimizer = optim.Adam([{'params': netBG.parameters()}, {'params': netBE.parameters()}], lr=lr, betas=(0.90, 0.999))

    netBE, netBG, complex_background = background_training_loop(args,netBE, netBG, optimizer,
                                                                                    dataset,
                                                                                    model_path, batch_size, device,
                                                                                    number_of_steps, evaluation_step)

    if complex_background: # if the background is complex, start new training with more complex model

        number_of_steps = max(args.n_complex, (len(dataset) // batch_size) * args.e_complex)
        evaluation_step = 1e10 # no evaluation

        netBE, netBG = utils.setup_background_models(device, dataset.image_height,dataset.image_width, complex_background)

        optimizer = optim.Adam([{'params': netBG.parameters()}, {'params': netBE.parameters()}], lr=lr,
                               betas=(0.90, 0.999))
        netBE, netBG, _ = background_training_loop(args,netBE, netBG, optimizer,
                                                                                        dataset, model_path,
                                                                                        batch_size, device,
                                                                                        number_of_steps,
                                                                                        evaluation_step)

    return netBE, netBG



