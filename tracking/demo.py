import _init_paths
import os
import random
import argparse
import cv2
import torch
import torch.utils.data
import numpy as np
from easydict import EasyDict as edict
import lib.models.model as model
from lib.utils.utils import (load_yaml,
                             load_pretrain,
                             cxy_wh_2_rect,
                             get_frames)
from lib.tracker.lighttrack import Lighttrack

torch.set_num_threads(1)
# if 'DISPLAY' not in os.environ:
#     os.environ['DISPLAY'] = 'localhost:12.0'


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='LightTrack Demo')
    parser.add_argument('--cfg', type=str, default='../experiments/LightTrackM.yaml', help='yaml configure file name')
    parser.add_argument('--resume',  default=None, help='resume checkpoint')
    parser.add_argument('--video', default='', type=str, help='videos or image files')

    parser.add_argument('--arch', dest='arch', help='backbone architecture')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--stride', type=int, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='NULL')

    args = parser.parse_args()
    return args


def track(inputs):
    video_player = inputs['player']
    siam_tracker = inputs['tracker']
    siam_net = inputs['network']
    args = inputs['args']
    config = inputs['config']
    start_frame, lost, boxes, toc = 0, 0, [], 0
    cv2.namedWindow('Demo', cv2.WND_PROP_FULLSCREEN)

    with torch.no_grad():
        for count, im in enumerate(video_player):
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

            tic = cv2.getTickCount()
            if count == start_frame:  # init
                # initialize video writer
                writer = cv2.VideoWriter('../demo/' + args.video_name + '_result' + '.avi',
                                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                         30,
                                         (im.shape[1], im.shape[0]))
                # initialize video tracker
                try:
                    init_rect = cv2.selectROI('Demo', im, False, False)
                except:
                    exit()
                cx = init_rect[0] + (init_rect[2] - 1) / 2  # center_x
                cy = init_rect[1] + (init_rect[3] - 1) / 2  # center_y
                w, h = init_rect[2], init_rect[3]
                target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                state = siam_tracker.init(im, target_pos, target_sz, siam_net)  # init tracker
                # write the first frame
                bbox = list(map(int, init_rect))
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
                writer.write(im)

            elif count > start_frame:  # tracking
                state = siam_tracker.track(state, im)
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                bbox = list(map(int, location))
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
                writer.write(im)
                cv2.imshow('Demo', im)
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break
            toc += cv2.getTickCount() - tic
    writer.release()
    cv2.destroyAllWindows()


def main():
    print('===> load config <====')
    args = parse_args()
    if args.cfg is not None:
        config = edict(load_yaml(args.cfg, subset=False))
        if config.COMMON.USE_CUDA:
            config.COMMON.DEVICE = torch.cuda.current_device()
        else:
            config.COMMON.DEVICE = 'cpu'
    else:
        raise Exception('Please set the config file for tracking test!')
    if args.video is not None:
        video_name = args.video.split('/')[-1].split('.')[0]
        args.video_name = video_name

    print('===> create Siamese model <====')
    # build tracker
    siam_tracker = Lighttrack(config)
    # build siamese network
    siam_net = model.__dict__[config.MODEL.ARCH](stride=config.MODEL.STRIDE)

    print('===> initialize Siamese model <====')
    siam_net = load_pretrain(siam_net, '../' + config.DEMO.RESUME)
    siam_net.eval()
    siam_net = siam_net.to(config.COMMON.DEVICE)

    print('===> init video player <====')
    video_player = get_frames(args.video)

    print('===> tracking! <====')
    inputs = {'player': video_player,
              'tracker': siam_tracker,
              'network': siam_net,
              'args': args,
              'config': config,
              }
    track(inputs)


if __name__ == '__main__':
    main()
