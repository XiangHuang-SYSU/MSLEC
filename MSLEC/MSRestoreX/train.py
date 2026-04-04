# flake8: noqa
import os.path as osp
import sys
sys.path.append('.')
from MSRestoreX.train_pipeline import train_pipeline

import MSRestoreX.archs
import MSRestoreX.data
import MSRestoreX.models
import MSRestoreX.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
