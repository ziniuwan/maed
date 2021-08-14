import os
import sys
sys.path.append('.')

import argparse
import numpy as np
import os.path as osp
from multiprocessing import Process, Pool
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from PIL import Image

from lib.core.config import INSTA_DIR, INSTA_IMG_DIR


def process_single_record(fname, outdir, split):
    sess = tf.Session()
    #print(fname)
    record_name = fname.split('/')[-1]
    for vid_idx, serialized_ex in enumerate(tf.python_io.tf_record_iterator(fname)):
        #print(vid_idx)
        os.makedirs(osp.join(outdir, split, record_name, str(vid_idx)), exist_ok=True)
        example = tf.train.Example()
        example.ParseFromString(serialized_ex)

        N = int(example.features.feature['meta/N'].int64_list.value[0])

        images_data = example.features.feature[
            'image/encoded'].bytes_list.value


        for i in range(N):
            image = np.expand_dims(sess.run(tf.image.decode_jpeg(images_data[i], channels=3)), axis=0)
            #video.append(image)
            image = Image.fromarray(np.squeeze(image, axis=0))
            image.save(osp.join(outdir, split, record_name, str(vid_idx), str(i)+".jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_dir', type=str, help='tfrecords file path', default=INSTA_DIR)
    parser.add_argument('--n', type=int, help='total num of workers')
    parser.add_argument('--i', type=int, help='current index of worker (from 0 to n-1)')
    parser.add_argument('--split', type=str, help='train or test')
    parser.add_argument('--out_dir', type=str, help='output images path', default=INSTA_IMG_DIR)
    args = parser.parse_args()

    fpaths = glob(f'{args.inp_dir}/{args.split}/*.tfrecord')
    fpaths = sorted(fpaths)
    
    total = len(fpaths)
    fpaths = fpaths[args.i*total//args.n : (args.i+1)*total//args.n]

    #print(fpaths)
    #print(len(fpaths))

    os.makedirs(args.out_dir, exist_ok=True)

    for idx, fp in enumerate(fpaths):
        process_single_record(fp, args.out_dir, args.split)