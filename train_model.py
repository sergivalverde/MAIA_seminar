# ----------------------------------------------------------
# NICMSLESIONS2
#
# Sergi Valverde 2019
# ----------------------------------------------------------

import os
import argparse
from base import train_model, test_model

# --------------------------------------------------
# set experimental parameters
# --------------------------------------------------
DATA_PATH = '/home/sergivalverde/DATA/mic/home/sergivalverde/DATA/VH_long'
TRAIN_IMAGE_ROOT = os.path.join(DATA_PATH)
TEST_IMAGE_ROOT = os.path.join(DATA_PATH)
# --------------------------------------------------

options = {}
options['training_path'] = os.path.join(TRAIN_IMAGE_ROOT, 'test')
options['test_path'] = os.path.join(TEST_IMAGE_ROOT, 'train')
options['input_data'] = ['flair_basal_brain.nii.gz',
                         'pd_basal_brain.nii.gz',
                         't1_basal_brain.nii.gz',
                         't2_basal_brain.nii.gz',
                         'flair_12m_brain.nii.gz',
                         'pd_12m_brain.nii.gz',
                         't1_12m_brain.nii.gz',
                         't2_12m_brain.nii.gz']

options['out_scan'] = 'lesionMask.nii.gz'
options['roi_mask'] = 'combined_wmh_mask_gm_0.nii.gz'
options['experiment'] = 'VHLONG2017_resunet32_hybrid_1000_s2'
options['use_attention'] = False
options['use_gpu'] = True
options['k-fold'] = 1

# computational resources
options['workers'] = 10
options['gpus'] = [1]

# other options
options['perc_training'] = 1
options['normalize'] = True
options['resample_epoch'] = False
options['data_augmentation'] = False
options['randomize_cases'] = True
options['input_channels'] = len(options['input_data'])
options['test_step'] = (16, 16, 16)
options['scale'] = 2
options['test_patch_shape'] = (32, 32, 32)
options['train_patch_shape'] = (32, 32, 32)
options['training_step'] = (1, 1, 1)
options['patch_threshold'] = 0.1
options['num_epochs'] = 200
options['batch_size'] = 32
options['train_split'] = 0.2
options['patience'] = 10
options['l_weight'] = 10
options['resume_training'] = False
options['sampling_type'] = 'hybrid'
options['num_pos_samples'] = 1000
options['min_sampling_th'] = 0.1
options['verbose'] = 1

if __name__ == "__main__":
    """
    main function
    """
    parser = argparse.ArgumentParser(description='vox2vox implementation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    opt = parser.parse_args()
    if opt.train:
        options['train'] = True
        train_model(options)
    else:
        options['train'] = False
        options['out_scan'] = options['roi_mask']
        options['sampling_type'] = 'balanced+roi'
        test_model(options)
