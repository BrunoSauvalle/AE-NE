import torch
import os
import stats
import main

def lasiesta_test(dataset_path,video_list):
    """ performs  foreground mask and backgroud generation for the LASIESTA videos listed in video_list,
    assuming the LASIESTA dataset is available at path dataset_path. computes and prints evaluation statistics """

    parser = main.create_parser()
    args = parser.parse_args()

    torch.cuda.empty_cache()

    messages = []

    for video_name in video_list:
        print(f'processing video {video_name}...')
        video_path = {}
        video_path['train_dataset'] = os.path.join(dataset_path, video_name, video_name)
        video_path['test_dataset'] = os.path.join(dataset_path, video_name, video_name)
        video_path['masks'] = os.path.join(dataset_path, video_name, 'results')
        video_path['backgrounds'] = os.path.join(dataset_path, video_name, 'backgrounds')
        video_path['models'] = os.path.join(dataset_path, video_name, 'models')
        video_path['GT'] = os.path.join(dataset_path, video_name, video_name + '-GT')

        main.compute_dynamic_backgrounds_and_masks(args,video_path)

        statistics = stats.compute_statistics('LASIESTA',video_name,video_path['masks'], video_path['GT'])
        print(statistics)
        messages.append(statistics)
    print(messages)



if __name__ == "__main__":

    # path to the dataset, to be updated
    dataset_path = '/workspace/nvme0n1p1/Datasets/LASIESTA'

    video_list = ['I_OC_01', 'I_OC_02']

    # uncomment to test on the full dataset
    # video_list = os.listdir(dataset_path)

    lasiesta_test(dataset_path, video_list)

