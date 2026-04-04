import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
from MSRestoreX.models import lr_scheduler as lr_scheduler

from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss

@MODEL_REGISTRY.register()
class MSRestoreXS1Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(MSRestoreXS1Model, self).__init__(opt)
        
    
    def setup_schedulers(self):
        """Set up schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer,
                                                    **train_opt['scheduler']))
        
        elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(
                        optimizer, **train_opt['scheduler']))
        
        else:
            raise NotImplementedError(
                f'Scheduler {scheduler_type} is not implemented yet.')
        
    '''
    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if "var" not in k or "vae" not in k:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    '''

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], broadcast_buffers = False, find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net
    
    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('clip_opt'):
            self.cri_clip = build_loss(train_opt['clip_opt']).to(self.device)
        else:
            self.cri_clip = None
        
        if train_opt.get('msvq_opt'):
            self.cri_vq = build_loss(train_opt['msvq_opt']).to(self.device)
        else:
            self.cri_vq = None
        
        if self.cri_pix is None and self.cri_clip is None:
            raise ValueError('Both pixel and SAP losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(MSRestoreXS1Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.gt)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.gt)
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output,_,_ = self.net_g(self.lq, self.gt)

        l_total = 0
        loss_dict = OrderedDict()
        #print("self.gt.shape", self.gt.shape)
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            #l_total = l_total + l_pix
            loss_dict['l_pix'] = l_pix
        
        if self.cri_vq:
            l_msvq, l_vq = self.cri_vq(self.output, self.gt)
            l_total += l_msvq
            l_total += l_vq
            loss_dict['l_msvq'] = l_msvq
            loss_dict['l_vq'] = l_vq
        #l_total.requires_grad_()
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
