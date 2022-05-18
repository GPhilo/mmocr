#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader

from mmocr.datasets import build_dataset

assert build_dataset is not None
import matplotlib.pyplot as plt
plt.ion()

def main():
    parser = argparse.ArgumentParser(description='Benchmark data loading')
    parser.add_argument('config', help='Train config file path.')
    parser.add_argument('--split', default='train', choices=['train', 'test'], help='Which pipeline to load')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    dataset = build_dataset(cfg.data[args.split])

    # prepare data loaders
    if 'imgs_per_gpu' in cfg.data:
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loader = build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        1,
        dist=False,
        seed=None)

    # im_fig, im_ax = plt.subplots()
    # im_fig.suptitle('Image')
    # im = None

    # Start progress bar after first 5 batches
    prog_bar = mmcv.ProgressBar(
        len(dataset) - 5 * cfg.data.samples_per_gpu, start=False)
    for i, data in enumerate(data_loader):
        if i == 5:
            prog_bar.start()
        for idx in range(len(data['img'])):
            if i >= 5:
                prog_bar.update()
            for batch_idx in range(len(data['img'].data[0])):
                # if im is None:
                #     im = im_ax.imshow((data['img'].data[idx][batch_idx].cpu().detach().numpy().transpose((1,2,0))*data['img_metas'].data[idx][batch_idx]['img_norm_cfg']['std']+data['img_metas'].data[idx][batch_idx]['img_norm_cfg']['mean'])/255)
                #     im = im_ax.imshow(data['img'].data[idx][batch_idx].cpu().detach().numpy().transpose((1,2,0)))
                # else:
                #     im.set_data((data['img'].data[idx][batch_idx].cpu().detach().numpy().transpose((1,2,0))*data['img_metas'].data[idx][batch_idx]['img_norm_cfg']['std']+data['img_metas'].data[idx][batch_idx]['img_norm_cfg']['mean'])/255)
                #     im.set_data(data['img'].data[idx][batch_idx].cpu().detach().numpy().transpose((1,2,0)))
                input('Press ENTER to continue...')
        


if __name__ == '__main__':
    main()
