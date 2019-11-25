import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from mri_utils.data_utils import get_voxel_coordenates
from mri_utils.processing import normalize_data


class MRI_DataPatchLoader(Dataset):
    """
    Data loader experiments

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 patch_size,
                 sampling_step,
                 random_pad=(0, 0, 0),
                 sampling_type='mask',
                 normalize=False,
                 min_sampling_th=0,
                 num_pos_samples=5000,
                 resample_epoch=False,
                 transform=None):
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - patch_size: patch size
        - sampling_step: sampling_step
        - sampling type: 'all: all voxels in input_mask,
                         'roi: all voxels in roi_mask,
                         'balanced: same number of positive and negative voxels
        - normalize: Normalize data (0 mean / 1 std)
        - min_sampling_th: Minimum value to extract samples (0 default)
        - num_pos_samples used when hybrid sampling
        - transform
        """

        self.patch_size = patch_size
        self.sampling_step = sampling_step
        self.random_pad = random_pad
        self.sampling_type = sampling_type
        self.patch_half = tuple([idx // 2 for idx in self.patch_size])
        self.normalize = normalize
        self.min_th = min_sampling_th
        self.resample_epoch = resample_epoch
        self.transform = transform
        self.num_pos_samples = num_pos_samples

        # preprocess scans

        # load MRI scans in memory
        self.input_scans, self.label_scans, self.roi_scans = self.load_scans(input_data,
                                                                             labels,
                                                                             rois,
                                                                             apply_padding=True)
        self.num_modalities = len(self.input_scans[0])
        self.input_train_dim = (self.num_modalities, ) + self.patch_size
        self.input_label_dim = (1, ) + self.patch_size

        # normalize scans if set update
        if normalize:
            self.input_scans = [[normalize_data(self.input_scans[i][m])
                                for m in range(self.num_modalities)]
                                for i in range(len(self.input_scans))]

        # Build the patch indexes based on the image index and the voxel
        # coordenates

        self.patch_indexes = self.generate_patch_indexes(self.roi_scans)

        print('> DATA: Training sample size:', len(self.patch_indexes))

    def __len__(self):
        """
        Get the legnth of the training set
        """
        return len(self.patch_indexes)

    def __getitem__(self, idx):
        """
        Get the next item. Resampling the entire dataset is considered if
        self.resample_epoch is set to True.
        """

        if idx == 0 and self.resample_epoch:
            self.patch_indexes = self.generate_patch_indexes(self.roi_scans)

        im_ = self.patch_indexes[idx][0]
        center = self.patch_indexes[idx][1]

        slice_ = [slice(c_idx-p_idx, c_idx+s_idx-p_idx)
                  for (c_idx, p_idx, s_idx) in zip(center,
                                                   self.patch_half,
                                                   self.patch_size)]

        # get current patches for both training data and labels
        input_train = np.stack([self.input_scans[im_][m][tuple(slice_)]
                                for m in range(self.num_modalities)], axis=0)
        input_label = np.expand_dims(
            self.label_scans[im_][0][tuple(slice_)], axis=0)

        # check dimensions and put zeros if necessary
        if input_train.shape != self.input_train_dim:
            print('error in patch', input_train.shape, self.input_train_dim)
            input_train = np.zeros(self.input_train_dim).astype('float32')
        if input_label.shape != self.input_label_dim:
            print('error in label')
            input_label = np.zeros(self.input_label_dim).astype('float32')

        if self.transform:
            input_train, input_label = self.transform([input_train,
                                                       input_label])

        return input_train, input_label

    def apply_padding(self, input_data, mode='constant', value=0):
        """
        Apply padding to edges in order to avoid overflow

        """
        padding = tuple((idx, size-idx)
                        for idx, size in zip(self.patch_half, self.patch_size))

        padded_image = np.pad(input_data,
                              padding,
                              mode=mode,
                              constant_values=value)
        return padded_image

    def load_scans(self,
                   input_data,
                   label_data,
                   roi_data,
                   apply_padding=True,
                   apply_canonical=False):
        """
        Applying padding to input scans. Loading simultaneously input data and
        labels in order to discard missing data in both sets.
        """

        input_scans = []
        label_scans = []
        roi_scans = []

        for s in input_data.keys():

            try:
                if apply_padding:
                    input_ = [self.apply_padding(nib.load(
                        input_data[s][i]).get_data().astype('float32'))
                              for i in range(len(input_data[s]))]
                    label_ = [self.apply_padding(nib.load(
                        label_data[s][i]).get_data().astype('float32'))
                              for i in range(len(label_data[s]))]
                    roi_ = [self.apply_padding(nib.load(
                        roi_data[s][i]).get_data().astype('float32'))
                              for i in range(len(roi_data[s]))]
                    input_scans.append(input_)
                    label_scans.append(label_)
                    roi_scans.append(roi_)
                    print('> DATA: Loaded scan', s,
                          'roi size:',  np.sum(roi_[0] > 0),
                          'label_size: ', np.sum(label_[0] > 0))
                else:
                    input_ = [(nib.load(
                        input_data[s][i]).get_data().astype('float32'))
                              for i in range(len(input_data[s]))]
                    label_ = [(nib.load(
                        label_data[s][i]).get_data().astype('float32'))
                              for i in range(len(label_data[s]))]
                    roi_ = [(nib.load(
                        roi_data[s][i]).get_data().astype('float32'))
                              for i in range(len(roi_data[s]))]
                    input_scans.append(input_)
                    label_scans.append(label_)
                    roi_scans.append(roi_)
                    print('> DATA: Loaded scan', s, 'roi size:',
                          np.sum(roi_[0] > 0))
            except:
                print('> DATA: Error loading scan', s, '... Discarding')

        return input_scans, label_scans, roi_scans

    def generate_patch_indexes(self, roi_scans):
        """
        Generate indexes to extract. Consider the sampling step and
        a initial random padding
        """
        training_indexes = []
        # patch_half = tuple([idx // 2 for idx in self.patch_size])
        for s, l, r, i in zip(self.input_scans,
                              self.label_scans,
                              roi_scans,
                              range(len(self.input_scans))):

            # sample candidates
            candidate_voxels = self.get_candidate_voxels(s[0], l[0], r[0])
            voxel_coords = get_voxel_coordenates(s[0],
                                                 candidate_voxels,
                                                 step_size=self.sampling_step,
                                                 random_pad=self.random_pad)
            training_indexes += [(i, tuple(v)) for v in voxel_coords]

        return training_indexes

    def get_candidate_voxels(self, input_mask, label_mask, roi_mask):
        """
        Sample input mask using different techniques:
        - all: extracts all voxels > 0 from the input_mask
        - mask: extracts all roi voxels
        - balanced: same number of positive and negative voxels from
                    the input_mask as defined by the roi mask
        - balanced+roi: same number of positive and negative voxels from
                    the roi and label mask

        - hybrid sampling:
          1. Set a number of positive samples == self.pos_samples
          2. Displace randomly its x, y, z position < self.patch_half
          3. Get the same number of negative samples from the roi mask
        """

        if self.sampling_type == 'image':
            sampled_mask = input_mask > 0

        if self.sampling_type == 'all':
            sampled_mask = input_mask > 0

        if self.sampling_type == 'mask':
            sampled_mask = roi_mask > 0

        if self.sampling_type == 'balanced':
            sampled_mask = label_mask > 0
            num_positive = np.sum(label_mask > 0)
            brain_voxels = np.stack(np.where(input_mask > self.min_th), axis=1)
            for voxel in np.random.permutation(brain_voxels)[:num_positive]:
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        if self.sampling_type == 'balanced+roi':
            sampled_mask = label_mask > 0
            num_positive = np.sum(label_mask > 0)
            roi_mask[label_mask == 1] = 0
            brain_voxels = np.stack(np.where(roi_mask > 0), axis=1)
            for voxel in np.random.permutation(brain_voxels)[:num_positive]:
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        if self.sampling_type == 'hybrid':
            x, y, z = np.where(label_mask > 0)
            number_of_samples = len(x)

            # sample voxels randomly until size equals self.num_samples
            if number_of_samples < self.num_pos_samples:
                expand_interval = int(self.num_pos_samples / number_of_samples) + 1
                x = np.repeat(x, expand_interval)
                y = np.repeat(y, expand_interval)
                z = np.repeat(z, expand_interval)

            index_perm = np.random.permutation(range(len(x)))
            x = x[index_perm][:self.num_pos_samples]
            y = y[index_perm][:self.num_pos_samples]
            z = z[index_perm][:self.num_pos_samples]

            # randomize the voxel center
            min_int_x = - self.patch_half[0] +1
            max_int_x = self.patch_half[0] -1
            min_int_y = - self.patch_half[1] +1
            max_int_y = self.patch_half[1] -1
            min_int_z = - self.patch_half[2] +1
            max_int_z = self.patch_half[2] -1
            x += np.random.randint(low=min_int_x,
                                   high=max_int_x,
                                    size=x.shape)
            y += np.random.randint(low=min_int_y,
                                   high=max_int_y,
                                   size=y.shape)
            z += np.random.randint(low=min_int_z,
                                   high=max_int_z,
                                   size=z.shape)

            # check boundaries
            x = np.maximum(self.patch_half[0], x)
            x = np.minimum(label_mask.shape[0] - self.patch_half[0], x)
            y = np.maximum(self.patch_half[1], y)
            y = np.minimum(label_mask.shape[1] - self.patch_half[1], y)
            z = np.maximum(self.patch_half[2], z)
            z = np.minimum(label_mask.shape[2] - self.patch_half[2], z)

            # assign the same number of positive and negative voxels
            sampled_mask = np.zeros_like(label_mask)

            # positive samples
            for x_v, y_v, z_v in zip(x, y, z):
                sampled_mask[x, y, z] = 1

            # negative samples
            brain_voxels = np.stack(np.where(roi_mask > 0), axis=1)
            for voxel in np.random.permutation(brain_voxels)[:self.num_pos_samples]:
                sampled_mask[voxel[0], voxel[1], voxel[2]] = 1

        return sampled_mask


class MRI_DataImageLoader(Dataset):
    """
    MRI Dataset for images

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 crop=True,
                 global_crop=True,
                 normalize=False,
                 transform=None):
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - crop only brain parts
        - normalize: Normalize data (0 mean / 1 std)
        - transform
        """
        self.crop = crop
        self.normalize = normalize
        self.transform = transform
        self.global_crop = global_crop
        # load MRI scans in memory
        self.input_scans, self.label_scans, self.roi_scans = self.load_scans(input_data,
                                                                             labels,
                                                                             rois)
        self.num_modalities = len(self.input_scans[0])

        # normalize scans if set update
        if normalize:
            self.input_scans = [[normalize_data(self.input_scans[i][m])
                                for m in range(self.num_modalities)]
                                for i in range(len(self.input_scans))]

        self.crop_coords = self.estimate_cropping_coords_dataset(self.input_scans)

        print('> DATA: sample size:', len(self.input_scans))

    def __len__(self):
        """
        Get the legnth of the training set
        """
        return len(self.input_scans)

    def __getitem__(self, idx):
        """
        Get the next item. Resampling the entire dataset is considered if
        self.resample_epoch is set to True.
        """

        input_images = self.input_scans[idx]
        label = self.label_scans[idx][0]
        roi = self.roi_scans[idx][0]

        if self.crop:

            if self.global_crop:
                crop_coords = self.crop_coords
            else:
                crop_coords = self.get_cropping_coords(roi)
            input_images = [image[crop_coords[0][0]:crop_coords[0][1],
                                  crop_coords[1][0]:crop_coords[1][1],
                                  crop_coords[2][0]:crop_coords[2][1]] for image in input_images]

            label = label[crop_coords[0][0]:crop_coords[0][1],
                          crop_coords[1][0]:crop_coords[1][1],
                          crop_coords[2][0]:crop_coords[2][1]]

            roi = roi[crop_coords[0][0]:crop_coords[0][1],
                      crop_coords[1][0]:crop_coords[1][1],
                      crop_coords[2][0]:crop_coords[2][1]]

        # resize images for training
        if len(input_images) > 1:
            input_train = np.stack(input_images, axis=0)
        else:
            input_train = np.expand_dims(input_images, axis=0)

        input_label = np.expand_dims(label, axis=0)

        return input_train, input_label, roi

    def apply_padding(self,
                      input_data,
                      pad_size=(4, 4),
                      mode='constant',
                      value=0):
        """
        Apply padding to edges in order to avoid overflow

        """
        padding = ((4, 4), (4, 4), (4, 4))
        padded_image = np.pad(input_data,
                              padding,
                              mode=mode,
                              constant_values=value)

        return padded_image

    def get_cropping_coords(self, input_mask):
        """
        Get the minimum/maximum coordenates for each plane x,y,z

        - input_mask contains the roi mask of the current image
        """

        x, y, z = np.where(input_mask > 0)
        return [(min(x), max(x)),
                (min(y), max(y)),
                (min(z), max(z))]

    def estimate_cropping_coords_dataset(self, input_mask):
        """
        Get the minimum/maximum coordenates for each plane x,y,z. Given that
        we are working with UNETS, we estimate an appropriate lenght for each
        image dimension such as dim_max - dim_min % 8

        - input_mask contains the roi mask of the current image
        """

        x_min, y_min, z_min = [], [], []
        x_max, y_max, z_max = [], [], []

        for input_mask in self.input_scans:
            x, y, z = np.where(input_mask[0] > 0)
            x_min.append(min(x))
            x_max.append(max(x))
            y_min.append(min(y))
            y_max.append(max(y))
            z_min.append(min(z))
            z_max.append(max(z))

        x_min, x_max = self.estimate_extra_dims(min(x_min),
                                                min(x_max),
                                                target=8)
        y_min, y_max = self.estimate_extra_dims(min(y_min),
                                                min(y_max),
                                                target=8)
        z_min, z_max = self.estimate_extra_dims(min(z_min),
                                                min(z_max),
                                                target=8)
        return [(x_min, x_max),
                (y_min, y_max),
                (z_min, z_max)]

    def estimate_extra_dims(self, input_min, input_max, target=8):
        """
        Estimate the extra dimensions needed for cropping on UNETS
        - ixonput_min: min_coord
        - input_max: max_coord
        """
        # find padding

        pad = target - ((input_max - input_min) % target)

        input_min -= pad // 2
        if pad % 2 == 0:
            input_max += pad // 2
        else:
            input_max += (pad // 2) + 1

        return input_min, input_max

    def load_scans(self, input_data, label_data, roi_data):
        """
        Applying padding to input scans. Loading simultaneously input data and
        labels in order to discard missing data in both sets.
        """

        input_scans = []
        label_scans = []
        roi_scans = []

        for s in input_data.keys():

            try:
                input_ = [self.apply_padding(nib.load(
                    input_data[s][i]).get_data().astype('float32'))
                          for i in range(len(input_data[s]))]
                label_ = [self.apply_padding(nib.load(
                    label_data[s][i]).get_data().astype('float32'))
                          for i in range(len(label_data[s]))]
                roi_ = [self.apply_padding(nib.load(
                    roi_data[s][i]).get_data().astype('float32'))
                        for i in range(len(roi_data[s]))]
                input_scans.append(input_)
                label_scans.append(label_)
                roi_scans.append(roi_)
                print('> DATA: Loaded scan', s, 'roi size:',
                      np.sum(roi_[0] > 0))
            except:
                print('> DATA: Error loading scan', s, '... Discarding')

        return input_scans, label_scans, roi_scans


class RotatePatch(object):
    """
    Patch rotation transformation: 90, 180 and 270 degrees
    """

    def __init__(self, rot_degree=90):
        self.rot_degree = rot_degree

    def __call__(self, v_sample):
        sample = v_sample[0].astype('float32')
        label = v_sample[1].astype('float32')

        if self.rot_degree == 90:
            t_sample = sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = label[:, :, ::-1, :].transpose(0, 1, 3, 2)

        if self.rot_degree == 180:
            t_sample = sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_sample = t_sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = label[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = t_label[:, :, ::-1, :].transpose(0, 1, 3, 2)

        if self.rot_degree == 270:
            t_sample = sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_sample = t_sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_sample = t_sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = label[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = t_label[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = t_label[:, :, ::-1, :].transpose(0, 1, 3, 2)

        t_sample = torch.from_numpy(t_sample.copy())
        t_label = torch.from_numpy(t_label.copy())

        return t_sample, t_label


class FlipPatch(object):
    """
    Patch flip transformation: horizontal and vertical flips
    """

    def __init__(self, flip_degree=0):
        self.flip_degree = flip_degree

    def __call__(self, v_sample):
        sample = v_sample[0].astype('float32')
        label = v_sample[1].astype('float32')

        if self.flip_degree == 0:
            t_sample = sample[:, :, :, ::-1]
            t_label = label[:, :, :, ::-1]

        if self.flip_degree == 180:
            t_sample = sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_sample = t_sample[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_sample = t_sample[:, :, :, ::-1]
            t_label = label[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = t_label[:, :, ::-1, :].transpose(0, 1, 3, 2)
            t_label = t_label[:, :, :, ::-1]

        t_sample = torch.from_numpy(t_sample.copy())
        t_label = torch.from_numpy(t_label.copy())

        return t_sample, t_label


class RandomHyper(object):
    """
    Random hyperintesities on WMH voxels
    self.max_interval controls the level of intensities levels allowed
    min_label sets the first label containing WMH intensities
    """

    def __init__(self, max_interval=3, min_label=30):
        self.max_interval = max_interval
        self.min_label = min_label

    def __call__(self, v_sample):
        sample = v_sample[0].astype('float32')
        label = v_sample[1].astype('float32')

        # randomize wmh labels for the particular patch
        sample[sample > self.min_label] += (np.random.randint(self.max_interval) + 1)
        sample = torch.from_numpy(sample.copy())
        label = torch.from_numpy(label.copy())
        return sample, label


class MRI_DataTestLoader(Dataset):
    """
    MRI Dataset for images.
    Test implementation

    """

    def __init__(self,
                 input_data,
                 labels,
                 rois,
                 crop=True,
                 global_crop=True,
                 normalize=False,
                 transform=None):
        """
        Arguments:
        - input_data: dict containing a list of inputs for each training scan
        - labels: dict containing a list of labels for each training scan
        - roi: dict containing a list of roi masks for each training scan
        - crop only brain parts
        - normalize: Normalize data (0 mean / 1 std)
        - transform
        """
        self.crop = crop
        self.normalize = normalize
        self.transform = transform
        self.global_crop = global_crop
        # load MRI scans in memory
        self.input_scans, self.label_scans, self.roi_scans = self.load_scans(input_data,
                                                                             labels,
                                                                             rois)
        self.num_modalities = len(self.input_scans[0])

        # normalize scans if set update
        if normalize:
            self.input_scans = [[normalize_data(self.input_scans[i][m])
                                for m in range(self.num_modalities)]
                                for i in range(len(self.input_scans))]

        self.crop_coords = self.estimate_cropping_coords_dataset(self.input_scans)

        print('> DATA: sample size:', len(self.input_scans))

    def __len__(self):
        """
        Get the legnth of the training set
        """
        return len(self.input_scans)

    def __getitem__(self, idx):
        """
        Get the next item. Resampling the entire dataset is considered if
        self.resample_epoch is set to True.
        """

        input_images = self.input_scans[idx]
        label = self.label_scans[idx][0]
        roi = self.roi_scans[idx][0]

        if self.crop:

            if self.global_crop:
                crop_coords = self.crop_coords
            else:
                crop_coords = self.get_cropping_coords(roi)
            input_images = [image[crop_coords[0][0]:crop_coords[0][1],
                                  crop_coords[1][0]:crop_coords[1][1],
                                  crop_coords[2][0]:crop_coords[2][1]] for image in input_images]

            label = label[crop_coords[0][0]:crop_coords[0][1],
                          crop_coords[1][0]:crop_coords[1][1],
                          crop_coords[2][0]:crop_coords[2][1]]

        # resize images for training
        if len(input_images) > 1:
            input_train = np.stack(input_images, axis=0)
        else:
            input_train = np.expand_dims(input_images, axis=0)

        input_label = np.expand_dims(label, axis=0)

        return input_train, input_label

    def apply_padding(self,
                      input_data,
                      pad_size=(4, 4),
                      mode='constant',
                      value=0):
        """
        Apply padding to edges in order to avoid overflow

        """
        padding = ((4, 4), (4, 4), (4, 4))
        padded_image = np.pad(input_data,
                              padding,
                              mode=mode,
                              constant_values=value)
        return padded_image

    def get_cropping_coords(self, input_mask):
        """
        Get the minimum/maximum coordenates for each plane x,y,z

        - input_mask contains the roi mask of the current image
        """

        x, y, z = np.where(input_mask > 0)
        return [(min(x), max(x)),
                (min(y), max(y)),
                (min(z), max(z))]

    def estimate_cropping_coords_dataset(self, input_mask):
        """
        Get the minimum/maximum coordenates for each plane x,y,z. Given that
        we are working with UNETS, we estimate an appropriate lenght for each
        image dimension such as dim_max - dim_min % 8

        - input_mask contains the roi mask of the current image
        """

        x_min, y_min, z_min = [], [], []
        x_max, y_max, z_max = [], [], []

        for input_mask in self.input_scans:
            x, y, z = np.where(input_mask[0] > 0)
            x_min.append(min(x))
            x_max.append(max(x))
            y_min.append(min(y))
            y_max.append(max(y))
            z_min.append(min(z))
            z_max.append(max(z))

        x_min, x_max = self.estimate_extra_dims(min(x_min),
                                                max(x_min),
                                                target=8)
        y_min, y_max = self.estimate_extra_dims(min(y_min),
                                                max(y_min),
                                                target=8)
        z_min, z_max = self.estimate_extra_dims(min(z_min),
                                                max(z_min),
                                                target=8)
        return [(x_min, x_max),
                (y_min, y_max),
                (z_min, z_max)]

    def estimate_extra_dims(input_min, input_max, target=8):
        """
        Estimate the extra dimensions needed for cropping on UNETS
        - input_min: min_coord
        - input_max: max_coord
        """
        # find padding
        pad = target - ((input_max - input_min) % target)

        input_min -= pad // 2
        if pad % 2 == 0:
            input_max += pad // 2
        else:
            input_max += (pad // 2) + 1

        return input_min, input_max

    def load_scans(self, input_data, label_data, roi_data):
        """
        Applying padding to input scans. Loading simultaneously input data and
        labels in order to discard missing data in both sets.
        """

        input_scans = []
        label_scans = []
        roi_scans = []

        for s in input_data.keys():

            try:
                input_ = [self.apply_padding(nib.load(
                    input_data[s][i]).get_data().astype('float32'))
                          for i in range(len(input_data[s]))]
                label_ = [self.apply_padding(nib.load(
                    label_data[s][i]).get_data().astype('float32'))
                          for i in range(len(label_data[s]))]
                roi_ = [self.apply_padding(nib.load(
                    roi_data[s][i]).get_data().astype('float32'))
                        for i in range(len(roi_data[s]))]
                input_scans.append(input_)
                label_scans.append(label_)
                roi_scans.append(roi_)
                print('> DATA: Loaded scan', s, 'roi size:',
                      np.sum(roi_[0] > 0))
            except:
                print('> DATA: Error loading scan', s, '... Discarding')

        return input_scans, label_scans, roi_scans
