from __future__ import print_function
import argparse
from distutils import command
from itertools import count
from sqlite3 import Cursor
import cv2
import lmdb
import numpy as np
import os
from os.path import exists,join

def  decode_img_from_str(img_str):
    img_arr=np.fromstring(img_str,np.uint8)
    img=cv2.imdecode(img_arr,cv2.IMREAD_COLOR)
    return img


def read_lmdb(lmdb_path):
    env=lmdb.open(lmdb_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (lmdb_path))
    with env.begin(write=False) as txn:
        num_samples=int(txn.get('num-samples'.endode()))
        
        for index in range(num_samples):
            sample_key='{:0>10d}'.format(index).encode('ascii')
            img_key=sample_key+b'_image'
            label_key=sample_key+b'_label'

            img_buffer=txn.get(img_key)
            buffer_dat=np.frombuffer(img_buffer,dtype=np.uint8)
            height,width,channel=32,400,3
            img=buffer_dat.reshape(height,width,channel)
            label=txn.get(label_key)
            yield img,int(label)

lmdb_path = '/ssdata/user/chang.chen/lmdb_data/base64_cut'
env = lmdb.open(lmdb_path, map_size)
txn = env.begin()
