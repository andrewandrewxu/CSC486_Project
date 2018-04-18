# -*- coding: utf-8 -*-
import os
import random
import cv2
import h5py
import numpy as np


class H5Dataset(object):
    def __init__(self, opts, type='train'):
        assert type == 'train' or type == 'valid', 'H5Dataset type must be one of \'train\' or \'valid\''
        if type == 'train':
            self._train_init(opts)
        else:
            self._valid_init(opts)
        print('Total number of images to sample: ', self.n_samples)

        self.max_frames = opts.max_frames
        self.np = opts.np
        self.input_res = opts.input_res
        self.img_list = opts.train_image_list if type == 'train' else opts.valid_image_list
        self.image_id_to_path_map = self._get_image_id_to_path_map()
        self.subtract_mean = not opts.no_subtract_mean

    def _train_init(self, opts):
        self.h5_path = os.path.join(opts.data_dir, opts.train_h5_path)
        self.h5_data = h5py.File(self.h5_path)
        self.n_samples = opts.nt_iters if opts.nt_iters else len(list(self.h5_data['image'].keys()))
        self.transforms = None # TODO: Create image transforms util file

    def _valid_init(self, opts):
        self.h5_path = os.path.join(opts.data_dir, opts.valid_h5_path)
        self.h5_data = h5py.File(self.h5_path)
        self.n_samples = opts.nv_iters if opts.nv_iters else len(list(self.h5_data['image'].keys()))
        self.transforms = None # TODO: Create image transforms util file

    def _get_image_id_to_path_map(self):
        get_path = lambda line: line.split()[0]
        get_id = lambda line: int(get_path(line).split(os.sep)[-1][:-4])  # Strip an image path to get the image id
        with open(self.img_list) as f:
            lines = f.readlines()
        return {get_id(l): get_path(l) for l in lines}

    def get(self, index=None):
        """
        If :index: is None, choose a random index.
        Returns a tuple of (images, labels) from the index
        """
        if not index:
            index = random.randint(1, self.n_samples)

        start = index
        end = index + self.max_frames - 1

        if end > self.n_samples:
            end = self.n_samples

        for i in range(start, end):
            try:
                indice = self.h5_data['indice'][str(index)][:][0]
            except KeyError:
                print('Warning: Could not open indice ', index, '.\nSkipping.')
                continue
            if indice < 0.5:
                end = i - 1
                break

        sequence_length = end - start + 1

        images = []
        labels = []

        for i in range(sequence_length):
            try:
                img_id = self.h5_data['image'][str(start+i)][:][0]
                img = cv2.imread(self.image_id_to_path_map[img_id+1])
            except KeyError:
                print('Warning: Could not open image id ', start+i, '.\nSkipping.')
                continue
            sized = cv2.resize(img, (368, 368))

            try:
                label = self.h5_data['label'][str(index+i)][:]
            except KeyError:
                print('Warning: Could not open label ', index+i, '.\nSkipping.')
                continue

            images.append(sized)
            labels.append(label)
        return np.array(images), np.array(labels)