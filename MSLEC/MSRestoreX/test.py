# flake8: noqa
import os.path as osp
import sys
sys.path.append('.')
from basicsr.test import test_pipeline

import MSRestoreX.archs
import MSRestoreX.data
import MSRestoreX.models
import time
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print("strat time-------------------")
    start_time = time.time()
    test_pipeline(root_path)
    end_time = time.time()
    run_time = end_time - start_time
    print("run time-----------------------------------------------------------------------:", run_time)
