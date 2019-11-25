import os
import shutil
import time
import nibabel as nib
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from mri_utils.data_utils import reconstruct_image, extract_patches
from mri_utils.data_utils import get_voxel_coordenates
from mri_utils.processing import normalize_data
from model import LesionNet
from dataset import MRI_DataPatchLoader
from dataset import RotatePatch, FlipPatch
from pyfiglet import Figlet


def train_model(options):
    """
    Train model
    """

    training_scans = sorted(os.listdir(options['training_path']))
    training_scans = training_scans[:int(len(training_scans) * options['perc_training'])]

    if options['randomize_cases']:
        random.shuffle(training_scans)

    # load training / validation data
    t_delimiter = int(len(training_scans) * (1 - options['train_split']))
    training_data = training_scans[:t_delimiter]
    validation_data = training_scans[t_delimiter:]

    # process data before training
    # - move images to the canonical space
    image_sequences = options['input_data'] + \
        [options['roi_mask']] + \
        [options['out_scan']]

    for scan in training_scans:
        transform_input_images(os.path.join(options['training_path'],
                                            scan),
                               image_sequences)

    print('--------------------------------------------------')
    print('TRAINING DATA:')
    print('--------------------------------------------------')

    input_data = {scan: [os.path.join(options['training_path'],
                                      scan,
                                      'tmp',
                                      d)
                         for d in options['input_data']]
                  for scan in training_data}
    labels = {scan: [os.path.join(options['training_path'],
                                  scan,
                                  'tmp',
                                  options['out_scan'])]
              for scan in training_data}
    rois = {scan: [os.path.join(options['training_path'],
                                scan,
                                'tmp',
                                options['roi_mask'])]
            for scan in training_data}

    # data augmentation
    set_transforms = None
    transform = []
    if options['data_augmentation']:
        transform += [RotatePatch(90),
                      RotatePatch(180),
p                      FlipPatch(0),
                      FlipPatch(180)]

    if len(transform) > 0:
        set_transforms = transforms.RandomChoice(transform)

    # dataset
    training_dataset = MRI_DataPatchLoader(input_data,
                                           labels,
                                           rois,
                                           patch_size=options['train_patch_shape'],
                                           sampling_step=options['training_step'],
                                           sampling_type=options['sampling_type'],
                                           normalize=options['normalize'],
                                           num_pos_samples=options['num_pos_samples'],
                                           transform=set_transforms)
    t_dataloader = DataLoader(training_dataset,
                              batch_size=options['batch_size'],
                              shuffle=True,
                              num_workers=options['workers'])

    print('--------------------------------------------------')
    print('VALIDATION DATA:')
    print('--------------------------------------------------')

    input_data = {scan: [os.path.join(options['training_path'], scan,
                                      'tmp',
                                      d)
                         for d in options['input_data']]
                  for scan in validation_data}
    labels = {scan: [os.path.join(options['training_path'],
                                  scan,
                                  'tmp',
                                  options['out_scan'])]
              for scan in validation_data}
    rois = {scan: [os.path.join(options['training_path'],
                                scan,
                                'tmp',
                                options['roi_mask'])]
            for scan in validation_data}

    validation_dataset = MRI_DataPatchLoader(input_data,
                                             labels,
                                             rois,
                                             patch_size=options['train_patch_shape'],
                                             sampling_step=options['training_step'],
                                             sampling_type=options['sampling_type'],
                                             normalize=options['normalize'],
                                             num_pos_samples=options['num_pos_samples'],
                                             transform=set_transforms)

    v_dataloader = DataLoader(validation_dataset,
                              batch_size=options['batch_size'],
                              shuffle=True,
                              num_workers=options['workers'])

    # model path
    script_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_folder, 'models')

    lesion_net = LesionNet(input_channels=options['input_channels'],
                           patch_shape=options['train_patch_shape'],
                           scale=options['scale'],
                           batch_size=options['batch_size'],
                           training_epochs=options['num_epochs'],
                           model_path=model_path,
                           patience=options['patience'],
                           use_attention=options['use_attention'],
                           model_name=options['experiment'],
                           gpu_mode=options['use_gpu'],
                           gpu_list=options['gpus'])

    # lesion_net.load_weights()
    lesion_net.train_model(t_dataloader, v_dataloader)


def train_k_fold_model(options):
    """
    Train model using a k-fold validation over a
    lispt of subjects
    """

    all_scans = sorted(os.listdir(options['training_path']))
    all_scans = all_scans[:int(len(all_scans) * options['perc_training'])]

    if options['randomize_cases']:
        random.shuffle(all_scans)

    kfold = options['k-fold']
    for k in range(0, len(all_scans), kfold):

        print('\n==========================================================')
        print('TRAINING K-FOLD', k)
        print('=========================================================\n\n')
        training_scans = all_scans[0:k] + all_scans[k+kfold:]
        testing_scans = all_scans[k:k+kfold]

        # load training / validation data
        t_delimiter = int(len(training_scans) * (1 - options['train_split']))
        training_data = training_scans[:t_delimiter]
        validation_data = training_scans[t_delimiter:]

        # process data before training
        # - move images to the canonical space
        image_sequences = options['input_data'] + \
            [options['roi_mask']] + \
            [options['out_scan']]

        for scan in training_scans:
            transform_input_images(os.path.join(options['training_path'],
                                                scan),
                                   image_sequences)

        print('--------------------------------------------------')
        print('TRAINING DATA:')
        print('--------------------------------------------------')

        input_data = {scan: [os.path.join(options['training_path'],
                                          scan,
                                          'tmp',
                                          d)
                             for d in options['input_data']]
                      for scan in training_data}
        labels = {scan: [os.path.join(options['training_path'],
                                      scan,
                                      'tmp',
                                      options['out_scan'])]
                  for scan in training_data}
        rois = {scan: [os.path.join(options['training_path'],
                                    scan,
                                    'tmp',
                                    options['roi_mask'])]
                for scan in training_data}

        # data augmentation
        set_transforms = None
        transform = []
        if options['data_augmentation']:
            transform += [RotatePatch(90),
                          RotatePatch(180),
                          FlipPatch(0),
                          FlipPatch(180)]

        if len(transform) > 0:
            set_transforms = transforms.RandomChoice(transform)

        # dataset
        training_dataset = MRI_DataPatchLoader(input_data,
                                               labels,
                                               rois,
                                               patch_size=options['train_patch_shape'],
                                               sampling_step=options['training_step'],
                                               sampling_type=options['sampling_type'],
                                               normalize=options['normalize'],
                                               num_pos_samples=options['num_pos_samples'],
                                               transform=set_transforms)
        t_dataloader = DataLoader(training_dataset,
                                  batch_size=options['batch_size'],
                                  shuffle=True,
                                  num_workers=options['workers'])

        print('--------------------------------------------------')
        print('VALIDATION DATA:')
        print('--------------------------------------------------')

        input_data = {scan: [os.path.join(options['training_path'], scan,
                                          'tmp',
                                          d)
                             for d in options['input_data']]
                      for scan in validation_data}
        labels = {scan: [os.path.join(options['training_path'],
                                      scan,
                                      'tmp',
                                      options['out_scan'])]
                  for scan in validation_data}
        rois = {scan: [os.path.join(options['training_path'],
                                    scan,
                                    'tmp',
                                    options['roi_mask'])]
                for scan in validation_data}

        validation_dataset = MRI_DataPatchLoader(input_data,
                                                 labels,
                                                 rois,
                                                 patch_size=options['train_patch_shape'],
                                                 sampling_step=options['training_step'],
                                                 sampling_type=options['sampling_type'],
                                                 normalize=options['normalize'],
                                                 num_pos_samples=options['num_pos_samples'],
                                                 transform=set_transforms)

        v_dataloader = DataLoader(validation_dataset,
                                  batch_size=options['batch_size'],
                                  shuffle=True,
                                  num_workers=options['workers'])

        # model path
        script_folder = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_folder, 'k_fold_models')
        k_fold_experiment = options['experiment'] + '_fold_' + str(k)

        lesion_net = LesionNet(input_channels=options['input_channels'],
                               patch_shape=options['train_patch_shape'],
                               scale=options['scale'],
                               batch_size=options['batch_size'],
                               training_epochs=options['num_epochs'],
                               model_path=model_path,
                               patience=options['patience'],
                               use_attention=options['use_attention'],
                               model_name=k_fold_experiment,
                               gpu_mode=options['use_gpu'],
                               gpu_list=options['gpus'])

        print('--------------------------------------------------')
        print('TEST_DATA:')
        print('--------------------------------------------------')
        for t in testing_scans:
            print('> DATA: Excluded scan:', t)

        # lesion_net.load_weights()
        lesion_net.train_model(t_dataloader, v_dataloader)

        # --------------------------------------------------
        # copy weights to test images
        # --------------------------------------------------
        trained_model_path = os.path.join(model_path, k_fold_experiment)
        for im_path in testing_scans:
            target_model_path = os.path.join(options['training_path'],
                                             im_path,
                                             'models')
            if not os.path.exists(target_model_path):
                os.mkdir(target_model_path)
            shutil.copy(trained_model_path,
                        os.path.join(target_model_path, options['experiment']))


def test_model(options):
    """
   Perform inference on several images (batch mode)
    Steps for each testing image:
    - Load bathches
    - Perform inference
    - Reconstruct the output image

    """
    # Lesion model
    lesion_net = LesionNet(input_channels=options['input_channels'],
                           patch_shape=options['train_patch_shape'],
                           model_name=options['experiment'],
                           scale=options['scale'],
                           gpu_mode=options['use_gpu'],
                           use_attention=options['use_attention'],
                           gpu_list=options['gpus'])

    if options['verbose']:
        show_info(options)

    print("Initializing model....")

    script_folder = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_folder, 'models')
    lesion_net.load_weights(model_path=model_path,
                            model_name=options['experiment'])


    patch_shape = options['test_patch_shape']
    step = options['test_step']
    image_path = options['test_path']
    image_sequences = options['input_data'] + \
        [options['roi_mask']] + \
        [options['out_scan']]

    # Perform inference for each of the images
    list_scans = sorted(os.listdir(image_path))
    for scan in list_scans:
        scan_path = os.path.join(image_path, scan)

        scan_time = time.time()

        # transform the scans before processing
        # - move to canonical space
        # - normalize between 0 and 1
        transform_input_images(scan_path, image_sequences)

        # get candidate voxels
        mask_image = nib.load(os.path.join(scan_path,
                                           'tmp',
                                           options['roi_mask']))
        ref_mask, ref_voxels = get_candidate_voxels(mask_image.get_data(),
                                                    step,
                                                    sel_method='all')

        # input images stacked as channels
        test_patches = get_data_channels(os.path.join(scan_path, 'tmp'),
                                         options['input_data'],
                                         ref_voxels,
                                         patch_shape,
                                         step,
                                         normalize=options['normalize'])

        print('--------------------------------------------------')
        print("Scan:", scan, "..... Predicting lesions")

        pred = lesion_net.test_net(test_patches)

        # reconstruction segmentation
        lesion_prob = reconstruct_image(np.squeeze(pred),
                                        ref_voxels,
                                        ref_mask.shape)

        # we transform computed images back to the T1 original space
        lesion_nifti = nib.Nifti1Image(lesion_prob.astype('<f4'),
                                       affine=mask_image.affine)

        orig_nifti = nib.load(os.path.join(scan_path,
                                           options['input_data'][0]))
        lesion_nifti = transform_canonical_to_orig(lesion_nifti,
                                                   orig_nifti)

        # save the results
        out_path = os.path.join(scan_path, options['experiment'])
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        lesion_nifti.to_filename(os.path.join(out_path, 'lesion_prob.nii.gz'))

        # shutil.rmtree(os.path.join(scan_path, 'tmp'))

        print('elapsed time', time.time() - scan_time)
        print('--------------------------------------------------')

def test_k_fold_model(options):
    """
   Perform inference on several images (batch mode)
    Steps for each testing image:
    - Load bathches
    - Perform inference
    - Reconstruct the output image

    """
    # Lesion model
    lesion_net = LesionNet(input_channels=options['input_channels'],
                           patch_shape=options['train_patch_shape'],
                           model_name=options['experiment'],
                           scale=options['scale'],
                           gpu_mode=options['use_gpu'],
                           gpu_list=options['gpus'])

    if options['verbose']:
        show_info(options)

    patch_shape = options['test_patch_shape']
    step = options['test_step']
    image_path = options['test_path']
    image_sequences = options['input_data'] + \
        [options['roi_mask']] + \
        [options['out_scan']]

    # Perform inference for each of the images
    list_scans = sorted(os.listdir(image_path))
    for scan in list_scans:
        scan_path = os.path.join(image_path, scan)
        model_path = os.path.join(scan_path, 'models')

        print("Initializing model....")
        lesion_net.load_weights(model_path=model_path,
                                model_name=options['experiment'])

        scan_time = time.time()

        # transform the scans before processing
        # - move to canonical space
        # - normalize between 0 and 1
        transform_input_images(scan_path, image_sequences)

        # get candidate voxels
        mask_image = nib.load(os.path.join(scan_path,
                                           'tmp',
                                           options['roi_mask']))
        ref_mask, ref_voxels = get_candidate_voxels(mask_image.get_data(),
                                                    step,
                                                    sel_method='all')

        # input images stacked as channels
        test_patches = get_data_channels(os.path.join(scan_path, 'tmp'),
                                         options['input_data'],
                                         ref_voxels,
                                         patch_shape,
                                         step,
                                         normalize=options['normalize'])

        print('--------------------------------------------------')
        print("Scan:", scan, "..... Predicting lesions")

        pred = lesion_net.test_net(test_patches)

        # reconstruction segmentation
        lesion_prob = reconstruct_image(np.squeeze(pred),
                                        ref_voxels,
                                        ref_mask.shape)

        # we transform computed images back to the T1 original space
        lesion_nifti = nib.Nifti1Image(lesion_prob.astype('<f4'),
                                       affine=mask_image.affine)

        orig_nifti = nib.load(os.path.join(scan_path,
                                           options['input_data'][0]))
        lesion_nifti = transform_canonical_to_orig(lesion_nifti,
                                                   orig_nifti)

        # save the results
        out_path = os.path.join(scan_path, options['experiment'])
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        lesion_nifti.to_filename(os.path.join(out_path, 'lesion_prob.nii.gz'))

        # shutil.rmtree(os.path.join(scan_path, 'tmp'))

        print('elapsed time', time.time() - scan_time)
        print('--------------------------------------------------')


def get_data_channels(image_path,
                      scan_names,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False):
    """
    Get data for each of the channels
    """
    out_patches = []
    for s in scan_names:
        current_scan = os.path.join(image_path, s)
        patches, _ = get_input_patches(current_scan,
                                       ref_voxels,
                                       patch_shape,
                                       step,
                                       normalize=normalize)
        out_patches.append(patches)

    return np.concatenate(out_patches, axis=1)


def get_input_patches(scan_path,
                      ref_voxels,
                      patch_shape,
                      step,
                      normalize=False,
                      expand_dims=True):
    """
    get current patches for a given scan
    """
    # current_scan = nib.as_closest_canonical(nib.load(scan_path)).get_data()
    current_scan = nib.load(scan_path).get_data()

    if normalize:
        current_scan = normalize_data(current_scan)

    patches, ref_voxels = extract_patches(current_scan,
                                          voxel_coords=ref_voxels,
                                          patch_size=patch_shape,
                                          step_size=step)

    if expand_dims:
        patches = np.expand_dims(patches, axis=1)

    return patches, ref_voxels


def get_candidate_voxels(input_mask,  step_size, sel_method='all'):
    """
    Extract candidate patches.
    """

    if sel_method == 'all':
        candidate_voxels = input_mask > 0

        voxel_coords = get_voxel_coordenates(input_mask,
                                             candidate_voxels,
                                             step_size=step_size)
    return candidate_voxels, voxel_coords


def transform_canonical_to_orig(canonical, original):
    """
    Transform back a nifti file that has been moved to the canonical space

    This function is a bit hacky, but so far it's the best way to deal with
    transformations between datasets without registration
    """

    ori2can = nib.io_orientation(original.affine)

    # transform the canonical image back to the original space
    ori2ori = nib.io_orientation(canonical.affine)
    can2ori = nib.orientations.ornt_transform(ori2ori, ori2can)
    return canonical.as_reoriented(can2ori)


def compute_pre_mask(T1_input, hist_bin=1):
    """
    Compute the ROI where brain intensities are (brain + skull).

    pre_mask = T1_input > min_intensity

    The minimum intensity is computed by taking the second bin in the histogram
    assuming the first one contains all the background parts

    input:
       T1_input: np.array containing the T1 image
       bin_edge: histogram bin number
    """

    hist, edges = np.histogram(T1_input, bins=64)
    pre_mask = T1_input > edges[hist_bin]

    return pre_mask


def transform_input_images(image_path, scan_names):
    """
    Transform input input images for processing
      + zero one normalization
      + move image sequences canonical space

    Images are stored in the tmp/ folder
    """

    # check if tmp folder is available
    tmp_folder = os.path.join(image_path, 'tmp')
    if os.path.exists(tmp_folder) is False:
        os.mkdir(tmp_folder)

    for s in scan_names:
        current_scan = os.path.join(image_path, s)
        nifti_orig = nib.load(current_scan)
        nifti_orig.get_data()[:] = normalize_data(nifti_orig.get_data(),
                                                  'zero_one')
        t1_nifti_canonical = nib.as_closest_canonical(nifti_orig)
        t1_nifti_canonical.to_filename(os.path.join(tmp_folder, s))


def show_info(options):
    """
    Show method information
    """
    f = Figlet(font="slant")
    print("--------------------------------------------------")
    print(f.renderText("NICMSLESIONS2"))
    print("NIC Multiple Scleroris Lesion Segmetnation v2")
    print("(c) Sergi Valverde, 2019")
    print(" ")
    print("version: v0.1")
    print("--------------------------------------------------")
    print(" ")
    print("Image information:")
    print("input path: ", options['test_path'])
    print("input image: ", options['input_data'][0])
    print("Output image: ", options['out_scan'])
    print("GPU using:", options['use_gpu'])
    print(" ")
    print("Model information")
    # print("Model path:", options['model_path'])
    print("Model name:", options['experiment'])
    print("--------------------------------------------------")
