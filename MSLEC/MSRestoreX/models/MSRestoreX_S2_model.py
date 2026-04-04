import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from MSRestoreX.models import lr_scheduler as lr_scheduler
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss
from thop import profile



@MODEL_REGISTRY.register()
class MSRestoreXS2Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(MSRestoreXS2Model, self).__init__(opt)
        
        self.net_g_S1 = build_network(opt['network_S1'])
        self.net_g_S1 = self.model_to_device(self.net_g_S1)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_S1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g_S1, load_path, True, param_key)
        
        self.net_g_S1.eval()
        if self.opt['dist']:
            #self.model_Es1 = self.net_g_S1.module.E_img
            self.model_s1_prior = self.net_g_S1.module.VAEPrior
        else:
            #self.model_Es1 = self.net_g_S1.E_img
            self.model_s1_prior = self.net_g_S1.VAEPrior
        
        if self.is_train:
            self.encoder_iter = opt["train"]["encoder_iter"]
            

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized in the second stage.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        parms=[]
        for k,v in self.net_g.named_parameters():
            #if "vae_former" in k:
            #if "denoise" in k or "condition" in k:
            if "denoise" in k or "condition" in k:
                parms.append(v)
        self.optimizer_e = self.get_optimizer(optim_type, parms, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_e)

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
        
        if train_opt.get('prior1_opt'):
            self.cri_prior1 = build_loss(train_opt['prior1_opt']).to(self.device)
        else:
            self.cri_prior1 = None

        if train_opt.get('prior2_opt'):
            self.cri_prior2 = build_loss(train_opt['prior2_opt']).to(self.device)
        else:
            self.cri_prior2 = None
        
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
        super(MSRestoreXS2Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def pad_test(self, window_size):        
        # scale = self.opt.get('scale', 1)
        scale = 1
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        lq = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        gt = F.pad(self.gt, (0, mod_pad_w*scale, 0, mod_pad_h*scale), 'reflect')
        return lq,gt,mod_pad_h,mod_pad_w

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            #print(" have window_size-------------------------------------")
            lq,gt,mod_pad_h,mod_pad_w=self.pad_test(window_size)
        else:
            lq=self.lq
            gt=self.gt
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(lq)
            
            self.net_g.train()
        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]


    def optimize_parameters(self, current_iter):
        l_total = 0
        loss_dict = OrderedDict()
        
        S1_prior_fea, S1_prior_fea_img = self.model_s1_prior(self.gt)
        
        if current_iter < self.encoder_iter:
            self.optimizer_e.zero_grad()
            S2_prior_fea, S2_prior_list = self.net_g.module.diffusion(self.lq,S1_prior_fea)
            S2_prior_fea_img, S2_prior_list_img = self.net_g.module.diffusion_img(self.lq,S1_prior_fea_img)
            
            
            l_abs_ms = self.cri_prior1(S1_prior_fea, S2_prior_fea)
            l_total += l_abs_ms
            loss_dict['l_abs_ms'] = l_abs_ms

            l_abs_global = self.cri_prior2(S1_prior_fea_img, S2_prior_fea_img)
            l_total += l_abs_global
            loss_dict['l_abs_global'] = l_abs_global
            
            l_total.backward()
            self.optimizer_e.step()
        else:
            self.optimizer_g.zero_grad()
            self.output, S2_prior_fea, S2_prior_fea_img = self.net_g(self.lq,S1_prior_fea,S1_prior_fea_img)
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
            if self.cri_vq:
                l_msvq, l_vq = self.cri_vq(self.output, self.gt)
                l_total += l_msvq
                l_total += l_vq
                loss_dict['l_msvq'] = l_msvq
                loss_dict['l_vq'] = l_vq
            
            
            l_abs_ms = self.cri_prior1(S1_prior_fea, S2_prior_fea)
            l_total += l_abs_ms
            loss_dict['l_abs_ms'] = l_abs_ms

            l_abs_global = self.cri_prior2(S1_prior_fea_img, S2_prior_fea_img)
            l_total += l_abs_global
            loss_dict['l_abs_global'] = l_abs_global

            l_total.backward()
            self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)