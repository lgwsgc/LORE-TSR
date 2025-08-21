from __future__ import print_function
import argparse
from distutils import command
from itertools import count
from sqlite3 import Cursor
import cv2
import lmdb
import numpy
import six
import os
from PIL import Image
from os.path import exists,join

def export_images(db_path,out_dir):
    env=lmdb.open(db_path)
    count=0
    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        for index in range(nSamples):
            index += 1  # lmdb starts with 1
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            print(label)
            print(img.size)
            

if __name__=='__main__':
    lmdb_path = '/ssdata/user/chang.chen/lmdb_data/base64_cut'
    out_images = "./images"
    export_images(lmdb_path,out_images)
