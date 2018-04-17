# -*- coding: utf-8 -*-

import h5py


class H5Dataset(object):
    def __init__(self, opts, type='train'):
        assert(type == 'train' or type == 'valid', 'H5Dataset type must be one of \'train\' or \'valid\'')
        if type == 'train':
            self._train_init(opts)
        else:
            self._valid_init(opts)
        print('Total Number of Images: %s', self.n_samples)

        self.max_frames = opts.max_frames
        self.np = opts.np
        self.input_res = opts.input_res
        self.img_list = opts.train_image_list if type == 'train' else opts.valid_image_list
        self.subtract_mean = not opts.no_subtract_mean

    def _train_init(self, opts):
        self.h5_path = opts.train_h5_path
        self.h5_data = h5py.File(self.h5_path)
        self.n_samples = opts.nt_iters
        self.transforms = None # TODO: Create image transforms util file

    def _valid_init(self, opts):
        self.h5_path = opts.valid_h5_path
        self.h5_data = h5py.File(self.h5_path)
        self.n_samples = opts.nv_iters if opts.nv_iters else len(list(self.h5_data.keys()))
        self.transforms = None # TODO: Create image transforms util file
