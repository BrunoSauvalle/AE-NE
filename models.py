
import torch
import torch.nn as nn


def get_positional_encoding(h,w):
    positional_encoding = torch.zeros(1, 2, h, w)
    for i in range(h):
        for j in range(w):
                if h > 1:
                    positional_encoding[0, 0, i, j] = i / (h-1) - 0.5
                else:
                    positional_encoding[0, 0, i, j] = 1
                if w > 1:
                    positional_encoding[0, 1, i, j] = j / (w-1) - 0.5
                else:
                    positional_encoding[0, 1, i, j] = 1
    return positional_encoding

class upsample_convolutional_block(nn.Module):
    """ upsample block composed of transpose convolutional layer, groupnorm and CELU"""
    def __init__(self,image_shape, n_channel_in,n_channel_out,n_group_out,kernel_size,stride,
                 padding = (0,0),final_layer = False, bias = False, output_padding=(0, 0)):

        super().__init__()

        module_list = [nn.ConvTranspose2d(n_channel_in+2, n_channel_out, kernel_size, stride,
                padding = padding, bias = bias, output_padding = output_padding)]

        if final_layer == False:
            module_list.append(nn.GroupNorm(n_group_out, n_channel_out))
            module_list.append(nn.CELU())

        self.upsample = nn.Sequential(*module_list)
        self.positional_encoding = nn.Parameter(get_positional_encoding(image_shape[0],image_shape[1]), requires_grad=False)

    def forward(self, input):
            expanded_positional_encoding = self.positional_encoding.expand(input.shape[0],2,input.shape[2],input.shape[3] )
            full_input = torch.cat([input,expanded_positional_encoding], dim = 1 )
            return self.upsample(full_input)

class downsample_convolutional_block(nn.Module):
    """ downsample block composed of convolutional layer, groupnorm and CELU"""

    def __init__(self,image_shape,n_channel_in,n_channel_out,n_group_out,kernel_size,stride,
                 bias = True, padding = (0,0)):
        super().__init__()

        conv_layer = nn.Conv2d(n_channel_in+2 , n_channel_out, kernel_size, stride,
                               padding=padding, bias=bias)
        self.downsample = nn.Sequential(conv_layer,nn.GroupNorm(n_group_out, n_channel_out),nn.CELU())
        self.positional_encoding = nn.Parameter(get_positional_encoding(image_shape[0],image_shape[1]), requires_grad=False)

    def forward(self, input):
            expanded_positional_encoding = self.positional_encoding.expand(input.shape[0],2,input.shape[2],input.shape[3] )
            full_input = torch.cat([input,expanded_positional_encoding], dim = 1 )
            return self.downsample(full_input)

class Background_Encoder(nn.Module):

    def __init__(self,image_height,image_width,complexity = False):
        super().__init__()

        self.complexity = complexity
        self.h = image_height
        self.w = image_width

        max_dim = max(image_height,image_width)
        if max_dim < 200:
            print('image size not implemented : image is too small')
            exit(0)
        if max_dim > 1000:
            print('Warning: large image size. You may need to reduce batch size to avoid memory overflow '
                  'and adjust other hyperparameters such as learning rate and numbers of steps accordingly')


        image_shapes = [(image_height, image_width)] # list of image shapes for all layers of the encoder
        kers = []  # list  of kernel sizes for all layers
        strs = [] # list of strides for all layers
        pads = [] # list of paddings for all layers

        current_w = image_width
        current_h = image_height

        n_blocks = 0

        # dynamically computes number of blocks, kernels, strides and paddings from the values of h and w
        while current_w > 1 or current_h > 1: # a new block is necessary

            n_blocks += 1

            if current_w > 5:
                ker_w = 5
                str_w = 3
                pad_w = 2
            else:
                ker_w = current_w
                str_w = 1
                pad_w = 0

            if current_h > 5:
                ker_h = 5
                str_h = 3
                pad_h = 2
            else:
                ker_h = current_h
                str_h = 1
                pad_h = 0

            new_w = 1 + (current_w + 2 * pad_w - ker_w) // str_w
            new_h = 1 + (current_h + 2 * pad_h -  ker_h ) // str_h

            image_shapes.append((new_h,new_w))
            kers.append((ker_h, ker_w))
            strs.append((str_h, str_w))
            pads.append((pad_h, pad_w))

            current_h = new_h
            current_w = new_w

        # sets channel schedule according to complexity and number of blocks of the encoder
        if n_blocks == 5:
                if complexity == False:
                    nchs = [3, 64, 160, 160, 32, 16]
                else:
                    nchs = [3, 64, 160, 160, 16, 16]
        elif n_blocks == 6:

            if complexity == False:
                nchs = [3, 64, 160, 160,160, 32, 16]
            else:
                nchs = [3, 64, 160, 160, 160, 16, 16]

        elif n_blocks == 7:

            if complexity == False:
                nchs = [3, 64, 160, 160,160, 160, 32, 16]
            else:
                nchs = [3, 64, 160, 160,160,  160, 16, 16]
        else:
            print('image size not implemented in this model')
            exit(0)

        ngs = [max(1, nchs[i] // 16) for i in range(n_blocks+1)] # number of groups for group normalization


        self.background_encoder  = nn.Sequential(*[downsample_convolutional_block(image_shapes[i],nchs[i],
                                                                nchs[i+1], ngs[i+1], kers[i],
                                                                strs[i], bias=False, padding=pads[i])
                                                   for i in range(n_blocks)])
    def forward(self, images):
        batch_size = images.shape[0]
        background_latents = self.background_encoder(images)
        return background_latents.reshape(batch_size, 16)

class Background_Generator(nn.Module):

    def __init__(self, image_height,image_width, complexity = False):

        super().__init__()

        self.complexity = complexity
        self.h = image_height
        self.w = image_width

        image_shapes = [(image_height, image_width)]
        n_blocks = 0    # number of blocks
        kers = []       # kernel sizes
        strs = []       # strides
        pads = []       # paddings value
        output_pads = []            # output paddings value for transpose convolutions
        final_layers = [True]
        current_w = image_width
        current_h = image_height

        while current_w > 1 or current_h > 1:

            n_blocks += 1

            if current_w > 5:
                ker_w = 5
                str_w = 3
                pad_w = 2
            else:
                ker_w = current_w
                str_w = 1
                pad_w = 0

            if current_h > 5:
                ker_h = 5
                str_h = 3
                pad_h = 2
            else:
                ker_h = current_h
                str_h = 1
                pad_h = 0

            new_w = 1 + (current_w + 2 * pad_w - ker_w ) // str_w
            output_pad_w =  (current_w + 2 * pad_w - ker_w )% str_w

            new_h = 1 + (current_h + 2 * pad_h - ker_h) // str_h
            output_pad_h = (  current_h + 2 * pad_h - ker_h ) % str_h

            image_shapes.append((new_h,new_w))
            kers.append((ker_h, ker_w))
            strs.append((str_h, str_w))
            pads.append((pad_h, pad_w))
            output_pads.append((output_pad_h,output_pad_w))

            current_h = new_h
            current_w = new_w
            final_layers.append(False)

        # channel schedules for generator
        if n_blocks == 5:
            if complexity == False:
                nchs = [4,144,256,256,32,16]
            else:
                nchs = [4,144,640,640,16,16]

        elif n_blocks == 6:
            if complexity == False:
                nchs = [4, 144, 256, 512,256, 32, 16]
            else:
                nchs = [4, 144, 640, 1280, 640, 16, 16]

        elif n_blocks == 7:
            if complexity == False:
                nchs = [4, 144, 256, 512, 512, 256, 32, 16]
            else:
                nchs = [4, 144, 640, 1280,1280, 640, 16, 16]
        else:
            print('image size not implemented in this model')
            exit(0)

        ngs = [max(1, nchs[i] // 16) for i in range(n_blocks + 1)] # number of groups for group norm

        self.background_generator = nn.Sequential(*[upsample_convolutional_block(image_shapes[i+1],nchs[i+1],
                            nchs[i],ngs[i],kers[i],strs[i],padding = pads[i],
                            final_layer = final_layers[i],output_padding=output_pads[i]
                            ) for i in reversed(range(n_blocks))
                            ])

    def forward(self,background_latents):
            batch_size = background_latents.shape[0]
            x = self.background_generator(background_latents.reshape(batch_size, 16,1,1)) # output shape bs x 4 x h x w
            background_images = torch.sigmoid(x)*255
            return background_images # range 0-255

