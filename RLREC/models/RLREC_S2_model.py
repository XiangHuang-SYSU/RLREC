import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from RLREC.models import lr_scheduler as lr_scheduler
from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss




@MODEL_REGISTRY.register()
class RLRECS2Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RLRECS2Model, self).__init__(opt)
        
        self.net_g_S1 = build_network(opt['network_S1'])
        self.net_g_S1 = self.model_to_device(self.net_g_S1)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_S1', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g_S1, load_path, True, param_key)
        
        self.net_g_S1.eval()
        if self.opt['dist']:
            self.model_s1_priorEwA = self.net_g_S1.module.Edino_A
            self.model_s1_priorEwS = self.net_g_S1.module.Edino_S
        else:
            self.model_s1_priorEwA = self.net_g_S1.Edino_A
            self.model_s1_priorEwS = self.net_g_S1.Edino_S
        
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
        
        if train_opt.get('prior_opt1'):
            self.cri_prior1 = build_loss(train_opt['prior_opt1']).to(self.device)
        else:
            self.cri_prior1 = None

        if train_opt.get('prior_opt2'):
            self.cri_prior2 = build_loss(train_opt['prior_opt2']).to(self.device)
        else:
            self.cri_prior2 = None

        if train_opt.get('dino_opt'):
            self.cri_dino = build_loss(train_opt['dino_opt']).to(self.device)
        else:
            self.cri_dino = None

        if self.cri_pix is None and self.cri_dino is None:
            raise ValueError('Both pixel and DINO losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        #self.mask = data['mask'].to(self.device)
        self.lq_Structure_gt_Appearance = data['lq_Structure_gt_Appearance'].to(self.device)
        self.lq_Appearance_gt_Structure = data['lq_Appearance_gt_Structure'].to(self.device)
        
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(RLRECS2Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
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
        S1_prior_fea1 = self.model_s1_priorEwA(self.lq_Structure_gt_Appearance,self.gt)
        S1_prior_fea2 = self.model_s1_priorEwS(self.lq_Appearance_gt_Structure,self.gt)
        
        if current_iter < self.encoder_iter:
            self.optimizer_e.zero_grad()
            S2_prior_fea = self.net_g.module.diffusion(self.lq, S1_prior_fea1, S1_prior_fea2)
            S2_prior_fea_wA,S2_prior_fea_wS = S2_prior_fea.chunk(2, dim=1)

            l_abs_prior_wA = self.cri_prior1(S1_prior_fea1, S2_prior_fea_wA)
            l_total += l_abs_prior_wA
            loss_dict['l_abs_prior_wA'] = l_abs_prior_wA
            l_abs_prior_wS = self.cri_prior2(S1_prior_fea2, S2_prior_fea_wS)
            l_total += l_abs_prior_wS
            loss_dict['l_abs_prior_wS'] = l_abs_prior_wS
            
            
            l_total.backward()
            self.optimizer_e.step()
        else:
            self.optimizer_g.zero_grad()
            self.output, S2_prior_fea_wA, S2_prior_fea_wS = self.net_g(self.lq,S1_prior_fea1,S1_prior_fea2)
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            
            if self.cri_dino:
                l_dino_Appearance, l_dino_Structure = self.cri_dino(self.output, self.gt)
                l_total += l_dino_Appearance
                l_total += l_dino_Structure
                loss_dict['l_dino_Appearance'] = l_dino_Appearance
                loss_dict['l_dino_Structure'] = l_dino_Structure

            l_abs_prior_wA = self.cri_prior1(S1_prior_fea1, S2_prior_fea_wA)
            l_total += l_abs_prior_wA
            loss_dict['l_abs_prior_wA'] = l_abs_prior_wA
            l_abs_prior_wS = self.cri_prior2(S1_prior_fea2, S2_prior_fea_wS)
            l_total += l_abs_prior_wS
            loss_dict['l_abs_prior_wS'] = l_abs_prior_wS
            
            l_total.backward()
            self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)