# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp

import mmcv

from mmocr.datasets.pipelines.crop import warp_img
from mmocr.utils.fileio import list_to_file

# TODO: Code is shit. Rewrite as a generic coco_dataset_converter that takes an imag root and a json file using COCO-style detection annotations.


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

    ann_list, imgs_list = [], []
    for gt_file in os.listdir(gt_dir):
        ann_list.append(osp.join(gt_dir, gt_file))
        imgs_list.append(osp.join(img_dir, gt_file.replace('gt_', 'MPSC_').replace('.txt', '.jpg')))

    files = list(zip(sorted(imgs_list), sorted(ann_list)))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
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


def load_img_info(files):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    assert osp.basename(gt_file).split('.')[0].replace('gt_', 'MPSC_') == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    img_info = dict(
        file_name=osp.join(osp.basename(img_file)),
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
                segmentation=[segmentation],
                word=ann['text'])
            anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info


def generate_ann(root_path, dst_path, split, image_infos, preserve_vertical):
    """Generate cropped annotations and label txt file.

    Args:
        root_path (str): The root path of the dataset
        split (str): The split of dataset. Namely: training or test
        image_infos (list[dict]): A list of dicts of the img and
            annotation information
        preserve_vertical (bool): Whether to preserve vertical texts
    """
    if not osp.isabs(dst_path):
        dst_image_root = osp.join(root_path, dst_path, 'image', split)
    else:
        dst_image_root = dst_path
    if split == 'train':
        dst_label_file = osp.join(root_path, dst_path, 'train_label.txt')
    elif split == 'test':
        dst_label_file = osp.join(root_path, dst_path, 'test_label.txt')
    os.makedirs(dst_image_root, exist_ok=True)

    lines = []
    for image_info in image_infos:
        index = 1
        src_img_path = osp.join(root_path, 'image', split, image_info['file_name'])
        image = mmcv.imread(src_img_path)
        src_img_root = image_info['file_name'].split('.')[0]

        for anno in image_info['anno_info']:
            word = anno['word']
            # TODO: This crops images using the bbox, not the segmentation polygon.
            #       Needs to be fixed to use rotated bboxes instead -> change crop_img to warp_img
            dst_img = warp_img(image, anno['segmentation'][0])
            h, w, _ = dst_img.shape

            # Skip invalid annotations
            if min(dst_img.shape) == 0:
                continue
            # Skip vertical texts
            if not preserve_vertical and h / w > 2:
                continue

            dst_img_name = f'{src_img_root}_{index}.png'
            index += 1
            dst_img_path = osp.join(dst_image_root, dst_img_name)
            mmcv.imwrite(dst_img, dst_img_path)
            lines.append(f'{osp.basename(dst_image_root)}/{dst_img_name} '
                         f'{word}')
    list_to_file(dst_label_file, lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and test set of MPSC ')
    parser.add_argument('root_path', help='Root dir path of MPSC')
    parser.add_argument('--dst-ims-root', default='dst_imgs', help='Destination dir path of cropped images')
    parser.add_argument(
        '--preserve-vertical',
        help='Preserve samples containing vertical texts',
        action='store_true')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path

    for split in ['train', 'test']:
        print(f'Processing {split} set...')
        with mmcv.Timer(print_tmpl='It takes {}s to convert MPSC annotation'):
            files = collect_files(
                osp.join(root_path, 'image', split),
                osp.join(root_path, 'annotation', split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            generate_ann(root_path, args.dst_ims_root, split, image_infos, args.preserve_vertical)


if __name__ == '__main__':
    main()
