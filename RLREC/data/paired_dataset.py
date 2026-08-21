from torch.utils import data as data
from torchvision.transforms.functional import normalize

from RLREC.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)

from RLREC.utils import FileClient, imfrombytes, img2tensor, padding, padding_DP, imfrombytesDP
from basicsr.utils.registry import DATASET_REGISTRY
import ptwt, pywt, torch

@DATASET_REGISTRY.register()
class PairedDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            geometric_augs (bool): Use geometric augmentations.

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)
        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
            #img_gt = imfrombytes(img_bytes, float32=False)
            #img_gt = img_gt.astype(np.float32)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
            #img_lq = imfrombytes(img_bytes, float32=False)
            #img_lq = img_lq.astype(np.float32)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        '''
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            #print("padding img_lq.shape:",img_lq.shape) #(512, 512, 3)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            #print("paired img_lq.shape:",img_lq.shape) #(512, 512, 3)

            # flip, rotation augmentations
            if self.geometric_augs:
                img_gt, img_lq = random_augmentation(img_gt, img_lq)
                #print("augmentation img_lq.shape:",img_lq.shape) #(512, 512, 3)
        '''    
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        coefficients = ptwt.wavedec2(img_lq, pywt.Wavelet("haar"),level=1, mode="constant")
        cA_input,(cH,cV,cD) = coefficients
        cA_input = cA_input.permute(1,2,0)
        cH = cH.permute(1,2,0)
        cV = cV.permute(1,2,0)
        cD = cD.permute(1,2,0)

        mask1 = torch.all(cH == 0, axis=2)
        mask2 = torch.all(cV == 0, axis=2)
        mask3 = torch.all(cD == 0, axis=2)
        mask = torch.where(mask1 & mask2 & mask3, 0, 255)
        
        coefficients = ptwt.wavedec2(img_gt, pywt.Wavelet("haar"),level=1, mode="constant")
        cA_gt,(cH_gt,cV_gt,cD_gt) = coefficients
        cA_gt = cA_gt.permute(1,2,0)

        # 将mask转为布尔索引
        mask_bool = mask == 0   # True 表示 mask 为 0 的区域
        # 创建结果图像1: mask==0 来自 input，mask!=0 来自 gt
        result1 = torch.where(mask_bool[..., None], cA_input, cA_gt)
        # 创建结果图像2: mask==1 来自 input，mask==0 来自 gt
        result2 = torch.where(mask_bool[..., None], cA_gt, cA_input)

        result1 = result1.permute(2,0,1)
        result2 = result2.permute(2,0,1)
        #print("result1:", result1)
        #print("result2:", result2)
        mask_rgb = mask.unsqueeze(-1).repeat(1, 1, 3).permute(2,0,1)
        mask_rgb = mask_rgb / 255

        lq_Structure_gt_Appearance = result1
        lq_Appearance_gt_Structure = result2
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            print("normalize img_lq:", img_lq)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'mask': mask_rgb,
            'lq_Structure_gt_Appearance': lq_Structure_gt_Appearance,
            'lq_Appearance_gt_Structure': lq_Appearance_gt_Structure,
            
        }

    def __len__(self):
        return len(self.paths)

