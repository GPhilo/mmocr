# TODO: 
# 1) Load JSON file
# 2) Extract the im filename, bboxes and texts for each sample
# 3) warp-crop and save the images; write out the annotation

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from pathlib import Path
import xml.etree.ElementTree as ET
import math
from PIL import Image
import numpy as np
from mmocr.datasets.pipelines.crop import warp_img

import mmcv


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
    for img_file in os.listdir(img_dir):
        ann_path = osp.join(gt_dir, osp.splitext(img_file)[0] + '.xml')
        if os.path.exists(ann_path):
            ann_list.append(ann_path)
            imgs_list.append(osp.join(img_dir, img_file))

    files = list(zip(imgs_list, ann_list))
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
    assert osp.basename(gt_file).split('.')[0] == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    try:
        img_info = dict(
            file_name=osp.join(osp.basename(img_file)),
            height=img.shape[0],
            width=img.shape[1],
            segm_file=osp.join(osp.basename(gt_file)))
    except AttributeError:
        print(f'Skip broken img {img_file}')
        return None

    if osp.splitext(gt_file)[1] == '.xml':
        img_info = load_xml_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info


def load_xml_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following:
    <annotations>
    ...
        <!--One of:-->
        <object>
            <name>something</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>157</xmin>
                <ymin>294</ymin>
                <xmax>237</xmax>
                <ymax>357</ymax>
            </bndbox>
        </object>
        <object>
            <name>something else</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <robndbox>
                <xmin>157.1234</xmin>
                <ymin>294.1234</ymin>
                <xmax>237.1234</xmax>
                <ymax>357.1234</ymax>
                <angle>0.021263</angle> // Angle w.r.t. NEGATIVE Y axis, positive clockwise, between the axis and the box "up" direction
            </robndbox>
        </object>
    ...
    </annotations>

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    obj = ET.parse(gt_file)
    root = obj.getroot()
    anno_info = []
    for object in root.iter('object'):
        word = object.find('name').text
        iscrowd = 1 if len(word) == 0 else 0
        if object.find('bndbox') is not None:
            x1 = float(object.find('bndbox').find('xmin').text)
            y1 = float(object.find('bndbox').find('ymin').text)
            x2 = float(object.find('bndbox').find('xmax').text)
            y2 = float(object.find('bndbox').find('ymax').text)
            x = max(0, min(x1, x2))
            y = max(0, min(y1, y2))
            w, h = abs(x2 - x1), abs(y2 - y1)
            bbox = [x1, y1, w, h]
            segmentation = [x, y, x + w, y, x + w, y + h, x, y + h]
        elif object.find('robndbox') is not None:
            # Rotated bbox annotation
            cx = float(object.find('robndbox').find('cx').text)
            cy = float(object.find('robndbox').find('cy').text)
            w = float(object.find('robndbox').find('w').text)
            h = float(object.find('robndbox').find('h').text)
            angle = float(object.find('robndbox').find('angle').text)
            # Logic straight out of labelImg2's PascalVocReader.addRotatedShape()
            p0x,p0y = rotatePoint(cx, cy, cx - w/2, cy - h/2, -angle)
            p1x,p1y = rotatePoint(cx, cy, cx + w/2, cy - h/2, -angle)
            p2x,p2y = rotatePoint(cx, cy, cx + w/2, cy + h/2, -angle)
            p3x,p3y = rotatePoint(cx, cy, cx - w/2, cy + h/2, -angle)
            segmentation = [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y]
            minx = min(p0x, p1x, p2x, p3x)
            miny = min(p0y, p1y, p2y, p3y)
            maxx = max(p0x, p1x, p2x, p3x)
            maxy = max(p0y, p1y, p2y, p3y)
            bbox = [minx, miny, maxx-minx, maxy-miny]

        # NOTE: bbox is the actual bbox of the 4 points
        #       segmentation is the rotated bbox    
        anno = dict(
            iscrowd=iscrowd,
            category_id=1,
            bbox=bbox,
            area=w * h,
            segmentation=[segmentation])
        if object.find('extra') is not None:
            anno['text'] = object.find('extra').text

        anno_info.append(anno)

    img_info.update(anno_info=anno_info)

    return img_info

def rotatePoint(xc, yc, xp, yp, theta):        
    xoff = xp-xc
    yoff = yp-yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return xc+pResx,yc+pResy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of a PascalVOC-compatible dataset')
    parser.add_argument('root_path', help='Root dir path of the dataset')
    parser.add_argument('out_fname', help='Filename of the output annotations file')
    parser.add_argument('--out-root-path', default=None, help='Output root dir. Defaults to root_path/rec')
    parser.add_argument('--img-prefix', default='imgs', help='subfolder of root_path containing the images')
    parser.add_argument('--annos-prefix', default=None, help='subfolder of root_path containing the annotations for each image. Defaults to img-prefix.')
    parser.add_argument('--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    if args.out_root_path is None:
        args.out_root_path = osp.join(args.root_path, 'rec')
    if args.annos_prefix is None:
        args.annos_prefix = args.img_prefix
    return args


def store_annotations(imgs_path, out_root_path, annotations, out_path):
    imgs_path = Path(imgs_path)
    out_root_path = Path(out_root_path)
    out_root_path.mkdir(exist_ok=True, parents=True)
    with Path(out_path).open('w') as fp:
        for img_info in annotations:
            im_fn = img_info['file_name']
            im = np.asarray(Image.open(imgs_path/im_fn))
            for idx, ann in enumerate(img_info['anno_info']):
                crop = warp_img(im, ann['segmentation'][0])
                crop_fn = f'{Path(im_fn).stem}_{idx}.png'
                Image.fromarray(crop).save(out_root_path/crop_fn)
                fp.write(f"{crop_fn} {ann['text']}\n")

def main():
    args = parse_args()
    root_path = args.root_path
    out_path = args.out_root_path
    with mmcv.Timer(print_tmpl='It takes {}s to convert annotations'):
        files = collect_files(
            osp.join(root_path, args.img_prefix), osp.join(root_path, args.annos_prefix))
        image_infos = collect_annotations(files, nproc=args.nproc)
        store_annotations(
            osp.join(root_path, args.img_prefix),
            osp.join(out_path, args.img_prefix),
            list(filter(None, image_infos)),
            osp.join(out_path, args.out_fname))


if __name__ == '__main__':
    main()
