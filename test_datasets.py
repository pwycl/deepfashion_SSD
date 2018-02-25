from __future__ import print_function
import argparse
import random
import numpy as np
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from PIL import Image  

import matplotlib.pyplot as plt

import numpy as np
import os
import warnings
import xml.etree.ElementTree as ET

import chainer

# from chainercv.datasets.voc import voc_utils
# from chainercv.utils import read_image
import os
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e
import six


class DeepFashionBboxDataset(chainer.dataset.DatasetMixin):

    
    def __init__(self, bbox_file, category_file, root='.', dtype=np.float32,
                 label_dtype=np.int32):
        _check_pillow_availability()
        if isinstance(bbox_file,six.string_types) and isinstance(bbox_file,six.string_types) :
            with open(bbox_file) as b_f:
                bbox=[]
                for i,line in enumerate(b_f):
                    pair=line.strip().split()
                    if len(pair)!=5:
                        raise ValueError(
                            'invalid format at line {}'.format(i)
                            )
                    bbox.append([pair[2],pair[1],pair[4],pair[3]])
        self._category = chainer.datasets.LabeledImageDataset(category_file, root)
        self._bbox=np.array(bbox,dtype=np.float64)
        self._dtype=dtype
        self._label_dtype=label_dtype

    def __len__(self):
        return len(self._category)

    def get_example(self, i):
        image,label=self._category[i]
        bbox=self._bbox[i]
        # image=image[:,bbox[0]:bbox[2],bbox[1]:bbox[3]]
        bbox=bbox.reshape((1,-1))
        label=np.array((label,),dtype=np.int32)
        return image,bbox,label

def _check_pillow_availability():
    if not available:
        raise ImportError('PIL cannot be loaded. Install Pillow!\n'
                          'The actual import error is as follows:\n' +
                          str(_import_error))

def main():
    parser = argparse.ArgumentParser(description='Chainer DeepFashionBboxDataset example:')
    parser.add_argument('--image_label','-il', help='Path to training image-label list file')
    parser.add_argument('--bbox', help='Path to training bbox list file')
    parser.add_argument('--image_root', '-TR', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--class_labels', '-cl', type=int, default=50,
                        help='Number of class labels')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--loaderjob', '-j', type=int, default=5,
                help='Number of parallel data loading processes')

    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    
    train=DeepFashionBboxDataset(args.bbox,args.image_label, args.image_root)
    img, bbox, label=train[10]
    print(bbox.shape)  # (2, 4)
    print(label.shape)  # (2,)
    img=np.array(img,dtype=np.uint8)
    img=np.transpose(img,(1,2,0))
    print(img.shape)
    plt.imshow(img)
    plt.show()
    # vis_bbox(img, bbox, label)

if __name__ == '__main__':
    main()
