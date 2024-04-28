"""
Microscopy dataloader

Takes h5files are keys
Expected h5file format:
<file> ---> <key> ---> "/noisy_im"
                   |-> "/clean_im"

In time seried data the noisy_im is 4D
"""

import numpy as np
from torch.utils.data import Dataset

from utils import *

class MicroscopyDataset(Dataset):
    """
    Dataset for loading microscopy data from h5files.
    """

    def __init__(self, h5files, keys,
                    time_cutout=30, cutout_shape=[64, 64],
                    num_samples_per_file=1, rng=None, test=False,
                    per_scaling=False, im_value_scale=[0,4096],
                    valu_thres=0.002, area_thres=0.25,
                    time_scale=0):
        """
        Initilize the denoising dataset

        @args:
            - h5files (h5file list): list of h5files to load images from
            - keys (str list list): List of list of keys. For every h5file, a list of keys are provided
            - cutout_shape (2 tuple int): patch size. Defaults to [64, 64]
            - num_samples_per_file (int): number of patches to cut out per image
            - rng (np.random): preset the seeds for the deterministic results
            - test (bool): whether this dataset is for evaluating test set. Does not make a cutout if True
            - per_scaling (bool): whether to use percentile scaling or scale with static values
            - im_value_scale (2 tuple int): the value to scale with
            - valu_thres (float): threshold of pixel value between background and foreground
            - area_thres (float): percentage threshold of area that needs to be foreground
            - time_scale (int): the time scale for time experiments
                - == 0: not time scaled exp
                - > 0: use the given time scale for making the average image
                - < 0: use all the time scale for making the average image
        """

        self.keys = keys
        self.N_files = len(self.keys)

        self.time_cutout = time_cutout
        self.cutout_shape = cutout_shape

        self.num_samples_per_file = num_samples_per_file

        self.test = test
        self.per_scaling = per_scaling
        self.im_value_scale = im_value_scale
        self.valu_thres = valu_thres
        self.area_thres = area_thres
        self.time_scale = time_scale

        # ------------------------------------------------

        self.start_samples = np.zeros(self.N_files)
        self.end_samples = np.zeros(self.N_files)

        self.start_samples[0] = 0
        self.end_samples[0] = len(self.keys[0]) * num_samples_per_file

        if(rng is None):
            self.rng = np.random.default_rng(seed=np.random.randint(0, np.iinfo(np.int32).max))
        else:
            self.rng = rng

        for i in range(1, self.N_files):
            self.start_samples[i] = self.end_samples[i-1]
            self.end_samples[i] = num_samples_per_file*len(self.keys[i]) + self.start_samples[i]

        self.tiff_dict = {}
        for i, hfile in enumerate(h5files):
            self.tiff_dict[i] = {}

            for key in keys[i]:
                self.tiff_dict[i][key] = {}

                noisy_data = np.array(hfile[key+"/noisy_im"]).astype(np.float32)
                clean_data = np.array(hfile[key+"/clean_im"]).astype(np.float32)

                if per_scaling:
                    noisy_data = normalize_image(noisy_data, percentiles=(1.5, 99.5), clip=True)
                    clean_data = normalize_image(clean_data, percentiles=(1.5, 99.5), clip=True)
                else:
                    noisy_data = normalize_image(noisy_data, values=im_value_scale, clip=True)
                    clean_data = normalize_image(clean_data, values=im_value_scale, clip=True)

                self.tiff_dict[i][key]["noisy_im"] = noisy_data
                self.tiff_dict[i][key]["clean_im"] = clean_data

            print(f"--> finish loading {hfile}")

    def load_one_sample(self, h5file, key):
        """
        Loads one sample from the h5file and key pair for regular paired image
        """
        noisy_im = []
        clean_im = []

        # get the image
        noisy_data = self.tiff_dict[h5file][key]["noisy_im"]
        clean_data = self.tiff_dict[h5file][key]["clean_im"]

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im = torch.from_numpy(train_noise.astype(np.float32))
            clean_im = torch.from_numpy(clean_cutout.astype(np.float32))

            return noisy_im, clean_im, key

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data, rng = self.rng)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape

        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data, rng=self.rng)

            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im = torch.from_numpy(train_noise.astype(np.float32))
            clean_im = torch.from_numpy(clean_cutout.astype(np.float32))

        return noisy_im, clean_im, key

    def load_one_sample_timed(self, h5file, key):
        """
        Loads one sample from the h5file and key pair of timed experiment paired images
        """
        # get the image
        noisy_data = self.tiff_dict[h5file][key]["noisy_im"]
        clean_data = self.tiff_dict[h5file][key]["clean_im"]

        if self.time_scale > 0:
            noisy_data = np.average(noisy_data[:self.time_scale], axis=0)
        if self.time_scale < 0:
            scale = np.random.randint(1,noisy_data.shape[0]+1)
            noisy_data = np.average(noisy_data[:scale], axis=0)

        if self.test:
            noisy_cutout = noisy_data[:,np.newaxis,:,:]
            clean_cutout = clean_data[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im = torch.from_numpy(train_noise.astype(np.float32))
            clean_im = torch.from_numpy(clean_cutout.astype(np.float32))

            return noisy_im, clean_im, key

        # pad symmetrically if not enough images in the time dimension
        if noisy_data.shape[0] < self.time_cutout:
            noisy_data = np.pad(noisy_data, ((0,self.time_cutout - noisy_data.shape[0]),(0,0),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0,self.time_cutout - clean_data.shape[0]),(0,0),(0,0)), 'symmetric')

        # random flip
        noisy_data, clean_data = self.random_flip(noisy_data, clean_data, rng = self.rng)

        if noisy_data.shape[1] < self.cutout_shape[0]:
            noisy_data = np.pad(noisy_data, ((0, 0), (0,self.cutout_shape[0] - noisy_data.shape[1]),(0,0)), 'symmetric')
            clean_data = np.pad(clean_data, ((0, 0), (0,self.cutout_shape[0] - clean_data.shape[1]),(0,0)), 'symmetric')

        if noisy_data.shape[2] < self.cutout_shape[1]:
            noisy_data = np.pad(noisy_data, ((0,0), (0,0), (0,self.cutout_shape[1] - noisy_data.shape[2])), 'symmetric')
            clean_data = np.pad(clean_data, ((0,0), (0,0), (0,self.cutout_shape[1] - clean_data.shape[2])), 'symmetric')

        T, RO, E1 = noisy_data.shape

        if(RO>=self.cutout_shape[0] and E1>=self.cutout_shape[1]):
            # define a set of cut range
            s_x, s_y, s_t = self.get_cutout_range(noisy_data, rng=self.rng)

            noisy_cutout = self.do_cutout(noisy_data, s_x, s_y, s_t)[:,np.newaxis,:,:]
            clean_cutout = self.do_cutout(clean_data, s_x, s_y, s_t)[:,np.newaxis,:,:]

            train_noise = noisy_cutout

            noisy_im = torch.from_numpy(train_noise.astype(np.float32))
            clean_im = torch.from_numpy(clean_cutout.astype(np.float32))

        return noisy_im, clean_im, key

    def get_cutout_range(self, data, rng):
        """
        get the starting positions of cutouts
        """
        t, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        # initial_s_t = rng.integers(0, t - ct) if t>ct else 0
        # initial_s_x = rng.integers(0, x - cx) if x>cx else 0
        # initial_s_y = rng.integers(0, y - cy) if y>cy else 0

        s_t = np.random.randint(0, t - ct + 1)
        s_x = np.random.randint(0, x - cx + 1)
        s_y = np.random.randint(0, y - cy + 1)

        return s_x, s_y, s_t

    def do_cutout(self, data, s_x, s_y, s_t):
        """
        Cuts out the patches
        """
        T, x, y = data.shape
        ct = self.time_cutout
        cx, cy = self.cutout_shape

        if T < ct or y < cy or x < cx:
            raise RuntimeError(f"File is borken because {T} is less than {ct} or {x} is less than {cx} or {y} is less than {cy}")

        return data[s_t:s_t+ct, s_x:s_x+cx, s_y:s_y+cy]

    def random_flip(self, noisy, clean, rng = np.random):

        flip1 = rng.integers(0,2) > 0
        flip2 = rng.integers(0,2) > 0

        def flip(image):
            if image.ndim == 2:
                if flip1:
                    image = image[::-1,:].copy()

                if flip2:
                    image = image[:,::-1].copy()
            else:
                if flip1:
                    image = image[:,::-1,:].copy()

                if flip2:
                    image = image[:,:,::-1].copy()

            return image

        return flip(noisy), flip(clean)

    def find_sample(self, index):
        ind_file = 0

        for i in range(self.N_files):
            if(index>= self.start_samples[i] and index<self.end_samples[i]):
                ind_file = i
                ind_in_file = int(index - self.start_samples[i])//self.num_samples_per_file
                break

        return ind_file, ind_in_file

    def __len__(self):
        total_num_samples = 0
        for key in self.keys:
            total_num_samples += len(key)*self.num_samples_per_file
        return total_num_samples

    def __getitem__(self, idx):

        #print(f"{idx}")
        sample_list = []
        count_list = []
        found = False

        # iterate 10 times to find the best sample
        for i in range(10):
            ind_file, ind_in_file = self.find_sample(idx)
            if self.time_scale == 0:
                sample = self.load_one_sample(ind_file, self.keys[ind_file][ind_in_file])
            else:
                sample = self.load_one_sample_timed(ind_file, self.keys[ind_file][ind_in_file])

            # The foreground content check
            valu_score = torch.count_nonzero(sample[1] > self.valu_thres)
            area_score = self.area_thres * sample[1].numel()
            if (valu_score >= area_score):
                found = True
                break

            sample_list.append(sample)
            count_list.append(valu_score)

        if not found:
            sample = sample_list[count_list.index(max(count_list))]

        return sample
