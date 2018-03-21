# -*- coding: utf-8 -*-
import tftables


class H5Dataset(object):
    def __init__(self, options, dataset_path, filename, input_transform=None):
        self.loader = tftables.load_dataset(
            filename=filename,
            dataset_path=dataset_path,
            batch_size=options.batch_size,
            input_transform=input_transform
        )

    def dequeue(self):
        return self.loader.dequeue()
