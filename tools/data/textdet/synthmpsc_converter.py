# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os.path as osp
import glob

import mmcv

from mmocr.utils import convert_annotations


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list = glob.glob(osp.join(gt_dir,'*','*.txt'))
    imgs_list = [ gtp.replace(gt_dir, img_dir).replace('.txt', '.jpg') for gtp in ann_list ]

    files = list(zip(sorted(imgs_list), sorted(ann_list)))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, imgs_root, nproc=1):
    """Collect the annotation information.

    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(args):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(args, tuple)

    img_file, gt_file = args
    im_parts = img_file.split(osp.sep)
    gt_parts = gt_file.split(osp.sep)
    assert osp.join(*im_parts[-2:]).split('.')[0] == osp.join(*gt_parts[-2:]).split('.')[0], \
      f"{osp.join(*im_parts[-2:]).split('.')[0]} != {osp.join(*gt_parts[-2:]).split('.')[0]}"
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.join(*im_parts[-2:]),
        height=img.shape[0],
        width=img.shape[1])

    if osp.splitext(gt_file)[1] == '.txt':
        img_info = load_anno_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info

def parse_annotation_file(gt_file):
    """Convert the text-line file in a dict with the appropriate keys"""
    with open(gt_file, 'r') as fp:
        lines = fp.read().splitlines()
    words_lst = []
    for ln in lines:
        toks = ln.split(',')
        pts = list(map(int, toks[:8]))
        txt = ','.join(toks[8:])
        maxx = max(pts[::2])
        maxy = max(pts[1::2])
        minx = min(pts[::2])
        miny = min(pts[1::2])
        words_lst.append({
            'text': txt,
            'box': [minx, miny, maxx, maxy]
            })
    return {'form': [{'words': words_lst}]}
    

def load_anno_info(gt_file, img_info):
    """Collect the annotation information.

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """

    annotation = parse_annotation_file(gt_file)
    
    anno_info = []
    for form in annotation['form']:
        for ann in form['words']:

            iscrowd = 1 if len(ann['text']) == 0 else 0

            x1, y1, x2, y2 = ann['box']
            x = max(0, min(math.floor(x1), math.floor(x2)))
            y = max(0, min(math.floor(y1), math.floor(y2)))
            w, h = math.ceil(abs(x2 - x1)), math.ceil(abs(y2 - y1))
            bbox = [x, y, w, h]
            segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]

            anno = dict(
                iscrowd=iscrowd,
                category_id=1,
                bbox=bbox,
                area=w * h,
                segmentation=[segmentation])
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training set of SynthMPSC ')
    parser.add_argument('root_path', help='Root dir path of SynthMPSC')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    print(f'Processing set...')
    with mmcv.Timer(print_tmpl='It takes {}s to convert MPSC annotation'):
        files = collect_files(
            osp.join(root_path, 'gen_pictures'),
            osp.join(root_path, 'gen_labels'))
        image_infos = collect_annotations(files, osp.join(root_path, 'gen_pictures'), nproc=args.nproc)
        convert_annotations(
            image_infos, osp.join(root_path,
                                    'instances_all.json'))


if __name__ == '__main__':
    main()
