
import os
import time
import stats
import main

def CDnet_test(root_path, category_list):
    """ performs  foreground mask and backgroud generation for the categories listed in category_list,
    assuming the CDnet dataset is available in the directory root_path. computes and prints evaluation statistics """

    parser = main.create_parser()
    args = parser.parse_args()


    start_time = time.time()

    rootpaths = {}
    rootpaths['dataset'] = os.path.join(root_path, 'dataset')
    rootpaths['masks'] = os.path.join(root_path, 'results')
    rootpaths['backgrounds'] = os.path.join(root_path, 'backgrounds')
    rootpaths['models'] = os.path.join(root_path, 'models')

    for (k, path) in rootpaths.items():
        if not os.path.exists(path):
            os.mkdir(path)

    messages = []

    for category in category_list:

        category_paths = {k:os.path.join(path,category) for (k,path) in rootpaths.items()}

        for (k,path) in category_paths.items():
            if not os.path.exists(path):
                os.mkdir(path)

        video_names = [video_name for video_name in os.listdir(category_paths['dataset'])
                            if os.path.isdir(os.path.join(category_paths['dataset'],video_name)) ]

        for video_name in video_names:

            video_paths = {k: os.path.join(path, video_name) for (k, path) in category_paths.items()}

            video_paths['test_dataset'] = video_paths['dataset']
            video_paths['train_dataset'] = video_paths['dataset']
            video_paths['GT'] = os.path.join(video_paths['dataset'],'groundtruth')
            spatial_roi_path = os.path.join(video_paths['dataset'],"ROI.bmp")
            temporal_roi_path = os.path.join(video_paths['dataset'], 'temporalROI.txt')

            video_start_time = time.time()

            for (k, path) in video_paths.items():
                if not os.path.exists(path):
                    os.mkdir(path)

            print(f"processing {video_name}")
            main.compute_dynamic_backgrounds_and_masks(args, video_paths)

            video_end_time = time.time()
            print(f"video folder {video_paths['dataset']} processing finished, computation time {video_end_time-video_start_time}")

            statistics = stats.compute_statistics('CDnet', video_name, video_paths['masks'], video_paths['GT'], spatial_roi_path, temporal_roi_path)
            print(statistics)
            messages.append(statistics)
        message = f'end of category {category}'+ 400*' '
        print(message)

    end_time = time.time()
    print(f'computation time : {end_time - start_time}')
    print(messages)


    print(f"foreground masks are stored in the directory {rootpaths['masks']}\n"
          f" reconstructed backgrounds are stored in the directory {rootpaths['backgrounds']}")


if __name__ == "__main__":

    # path to the directory containing the dataset folder, to be updated
    root_path = '/workspace/nvme0n1p1/Datasets/CDnet2014'

    category_list = ['baseline']

    # uncomment the following to test on all categories
    # category_list = ['badWeather','thermal','cameraJitter','dynamicBackground',
    #                           'baseline', 'nightVideos', 'lowFramerate','intermittentObjectMotion','turbulence','shadow','PTZ']


    CDnet_test(root_path, category_list)
