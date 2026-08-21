import torch
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from RLREC.models import lr_scheduler as lr_scheduler

from basicsr.archs import build_network
from basicsr.utils import get_root_logger
from basicsr.losses import build_loss

@MODEL_REGISTRY.register()
class RLRECS1Model(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(RLRECS1Model, self).__init__(opt)
        
    
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
                
        if self.cri_pix is None:
            raise ValueError('Both pixel and SAP losses are None.')

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
        super(RLRECS1Model, self).nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.gt, self.lq_Structure_gt_Appearance, self.lq_Appearance_gt_Structure)
                
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.gt, self.lq_Structure_gt_Appearance, self.lq_Appearance_gt_Structure)
                
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output, _, _ = self.net_g(self.lq, self.gt, self.lq_Structure_gt_Appearance, self.lq_Appearance_gt_Structure)
        

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
