"""
Data Utilities used by CNNT

Provides loading and preparing data facilities
config is the borrowed from main.py


"""

import os
import h5py
import random
import logging

from microscopy_dataset import *


def load_data(config):
    """
    Defines how to load micro data

    @input:
        config: config file from main
    """

    ratio = [x/100 for x in config.ratio]

    h5files = []
    train_keys = []
    val_keys = []
    test_keys = []

    for file in config.h5files:
        if not os.path.exists(file):
            logging.info(f"File not found: {file}")
            exit(-1)

        try:
            logging.info(f"reading from file: {file}")
            h5file = h5py.File(file,libver='latest',mode='r')
            keys = list(h5file.keys())
        except:
            logging.info(f"Error reading file: {file}")
            exit(-1)

        n = len(keys)
        if config.fine_samples > 0:
            assert(len(config.h5files) == 1), f"Can only finetune with one train dataset"
            h5files.append(h5file)
            train_keys.append(keys[:config.fine_samples])
            val_keys.append(keys[-5:])
            test_keys.append(keys[-5:])

            break   # since only one file no need for rest
        random.shuffle((keys))

        h5files.append(h5file)
        train_keys.append(keys[:int(ratio[0]*n)])
        val_keys.append(keys[int(ratio[0]*n):int((ratio[0]+ratio[1])*n)])
        test_keys.append(keys[int((ratio[0]+ratio[1])*n):int((ratio[0]+ratio[1]+ratio[2])*n)])

        # make sure there is no empty testing
        if len(test_keys[-1])==0:
            test_keys[-1] = keys[-1:]
        if len(val_keys[-1])==0:
            val_keys[-1] = test_keys[-1]

    cutout_shape=[config.height[0], config.width[0]]
    cutout_shape_larger = [config.height[-1], config.width[-1]]

    train_set = []

    for hw in zip(config.height, config.width):
        train_set.append(MicroscopyDataset(h5files=h5files, keys=train_keys,
                                            time_cutout=config.time,
                                            cutout_shape=hw,
                                            num_samples_per_file=8,
                                            rng = None,
                                            per_scaling = config.per_scaling,
                                            im_value_scale = config.im_value_scale,
                                            valu_thres=config.valu_thres,
                                            area_thres=config.area_thres,
                                            time_scale=config.time_scale)
        )

    if config.train_only:
        val_set = []
        val_set_larger = []
        test_set = []
        test_set_larger = []
    else:
        val_set = MicroscopyDataset(h5files=h5files, keys=val_keys,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape,
                                        rng = None,
                                        per_scaling = config.per_scaling,
                                        im_value_scale = config.im_value_scale,
                                        valu_thres=config.valu_thres,
                                        area_thres=config.area_thres,
                                        time_scale = config.time_scale)

        val_set_larger = MicroscopyDataset(h5files=h5files, keys=val_keys,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape_larger,
                                        rng = None,
                                        per_scaling = config.per_scaling,
                                        im_value_scale = config.im_value_scale,
                                        valu_thres=config.valu_thres,
                                        area_thres=config.area_thres,
                                        time_scale = config.time_scale)

        test_set = MicroscopyDataset(h5files=h5files, keys=test_keys,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape,
                                        rng = None,
                                        per_scaling = config.per_scaling,
                                        im_value_scale = config.im_value_scale,
                                        valu_thres=config.valu_thres,
                                        area_thres=config.area_thres,
                                        time_scale = config.time_scale)

        test_set_larger = MicroscopyDataset(h5files=h5files, keys=test_keys,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape_larger,
                                        rng = None,
                                        per_scaling = config.per_scaling,
                                        im_value_scale = config.im_value_scale,
                                        valu_thres=config.valu_thres,
                                        area_thres=config.area_thres,
                                        time_scale = config.time_scale)

    if config.test_case != None:

        h5files = []
        keyss = []
        for file in config.test_case:

            try:
                logging.info(f"reading from file (test_case): {file}")
                h5file = h5py.File(file,libver='latest',mode='r')
                keys = list(h5file.keys())
            except:
                logging.info(f"Error reading file (test_case): {file}")
                exit(-1)

            h5files.append(h5file)
            keyss.append(keys)

        test_set = MicroscopyDataset(h5files=h5files, keys=keyss,
                                        time_cutout=config.time,
                                        cutout_shape=cutout_shape_larger,
                                        rng = None,
                                        per_scaling = config.per_scaling,
                                        im_value_scale = config.im_value_scale,
                                        valu_thres=config.valu_thres,
                                        area_thres=config.area_thres,
                                        time_scale = config.time_scale)
        test_set_larger = test_set

    return train_set, val_set, test_set, val_set_larger, test_set_larger
