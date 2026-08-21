# flake8: noqa
import os.path as osp
import sys
sys.path.append('.')
from RLREC.train_pipeline import train_pipeline

import RLREC.archs
import RLREC.data
import RLREC.models
import RLREC.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
