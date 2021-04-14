"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: main_root
"""
from os.path import join

main_root = './'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = join(main_root, 'data/dataset_extras')

# Path to test/train npz files
DATASET_FILES = [ {'3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train_protocol1.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz')
                  }
                ]


CUBE_PARTS_FILE = join(main_root, 'data/cube_parts.npy')
JOINT_REGRESSOR_TRAIN_EXTRA = join(main_root, 'data/J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = join(main_root, 'data/J_regressor_h36m.npy')
VERTEX_TEXTURE_FILE = join(main_root, 'data/vertex_texture.npy')
STATIC_FITS_DIR = join(main_root, 'data/static_fits')
SMPL_MEAN_PARAMS = join(main_root, 'data/smpl_mean_params.npz')
SMPL_MODEL_DIR = join(main_root, 'data/smpl')
DATASET_PKL_PATH = join(main_root, 'datasets_pkl')
PRE_MODEL_PATH = join(main_root, 'data/model_checkpoint.pt')
