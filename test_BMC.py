import torch
import os
import time
import stats
import main

def BMC2012_test(dataset_path, video_names_list):
    """ performs  foreground mask and backgroud generation for the BMC videos listed in video_names_list,
    assuming the BMC dataset is available at path dataset_path. computes and prints evaluation statistics """


    parser = main.create_parser()
    args = parser.parse_args()

    torch.cuda.empty_cache()

    messages = []
    for video_name in video_names_list:
        print(f'processing video {video_name}...')

        video_path = {}
        video_path['train_dataset'] = os.path.join(dataset_path, video_name, 'frames_png')
        video_path['test_dataset'] = os.path.join(dataset_path, video_name, 'color')
        video_path['masks'] = os.path.join(dataset_path, video_name, 'results')
        video_path['backgrounds'] = os.path.join(dataset_path, video_name, 'backgrounds')
        video_path['models'] = os.path.join(dataset_path, video_name, 'models')
        video_path['GT'] = os.path.join(dataset_path, video_name, 'private_truth')

        start_time  = time.time()

        main.compute_dynamic_backgrounds_and_masks(args, video_path)

        finish_time = time.time()
        messages.append(f'computation time was {finish_time - start_time}, starting evaluation...')
        statistics = stats.compute_statistics('BMC2012', video_name, video_path['masks'], video_path['GT'])
        print(statistics)
        messages.append(statistics)
    print(messages)

if __name__ == "__main__":

    # location of the dataset, to be updated
    dataset_path = '/workspace/nvme0n1p1/Datasets/BMC2012/real_videos'

    # frames should be extracted in png format from the video file provided with the dataset and stored
    # in subfolders in each video file,
    # for example, the folder "real_videos/video001" should have the following subfolders :
    # color : test dataset provided on the website
    # frames_png : frames extracted from the video in png format
    # private_truth : ground truth masks provided with the dataset
    # be careful that Video_007 has more test samples (110) than ground truth masks, which requires a manual cleaning of the file
    # evaluation is performed using usual definition of F-measure. To compare with other publiashed results, use the evaluation
    # tool provided with the BMC dataset

    video_names_list = ['Video_008']

    BMC2012_test(dataset_path, video_names_list)

