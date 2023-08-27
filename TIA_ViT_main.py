############################# TIA_ViT & Swin_Transformer_Demo ##############################
#### Author: Dr.Pan Huang
#### Email: panhuang@cqu.edu.cn
#### Department: COE, Chongqing University
#### Attempt: Testing Swin_Transformer & TIA_ViT model

########################## API Section #########################
import skimage.color
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from torchsummaryX import summary
from tensorboardX import SummaryWriter
from SIL_Model.SwinT_models.models.swin_transformer import SwinTransformer
from SIL_Model.SwinT_models.models.SwinT_model_modules import SwinT_Net, TIA_ViT,TIA_ViT_Ablation, TIA_ViT_Visulization, \
    creating_swinT
from SIL_Utils.fit_functions import Single_out_fit, searching_best_lr, testing_funnction, Multiple_out_fit
from SIL_Utils.ablation_experiments import save_model, acc_scores, to_np_category
import cv2
from skimage import io
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
import PIL
import seaborn as sns
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import to_pil_image
from SIL_Learning.Adaptive_Dual_Semantic_CAM import Adaptive_Dual_Semantic_CAM
from SIL_Learning.CAM import CAM
from SIL_Learning.Grad_CAM import Grad_CAM
from SIL_Learning.Grad_CAM_Plus2 import Grad_CAM_Plus2
from SIL_Learning.Score_CAM import Score_CAM
from SIL_Learning.Smooth_Grad_CAM_PP import Smooth_Grad_CAM_Plus2
from SIL_Learning.Layer_CAM import Layer_CAM
from SIL_Learning.XGrad_CAM import XGrad_CAM


sns.set(font='Times New Roman', font_scale=0.6)

########################## seed_function #########################
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):
    random.seed(7 + worker_id)
    np.random.seed(7 + worker_id)
    torch.manual_seed(7 + worker_id)
    torch.cuda.manual_seed(7 + worker_id)
    torch.cuda.manual_seed_all(7 + worker_id)

########################## main_function #########################
if __name__ == '__main__':
    ########################## Hyparameters #########################
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    ads_cam_cls_num = 2
    batch_size = 16
    epochs = 100

    ########################## reading datas and processing datas #########################
    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()
                                       , transforms.Normalize(mean=0.5, std=0.5)])
    train_data = ImageFolder(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_greece\Train',
                             transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_data = ImageFolder(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_greece\Test',
                            transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size , shuffle=False, num_workers=1)
    val_data = ImageFolder(r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_greece\Val',
                           transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=1)
    print('train_data:', '\n', train_data, '\n')

    ########################## creating models and visuling models #########################
    print('########################## creating models and visuling models #########################')
    swinT_base = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=class_num,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False)

    checkpoint = torch.load(
        r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\SwinT\swin_tiny_patch4_window7_224_22k.pth',
        map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = swinT_base.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            #logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = swinT_base.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            #logger.warning(f"Error in loading {k}, passing......")
            pass
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = swinT_base.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            #logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(swinT_base.head.bias, 0.)
            torch.nn.init.constant_(swinT_base.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']

    swinT_base.load_state_dict(state_dict, strict=False)

    nn.init.trunc_normal_(swinT_base.head.weight, std=.02)
    print(swinT_base.layers[0].blocks[0].mlp.fc2.weight)

    ### creating a SwinT model
    #swinT_net = SwinT_Net(base_model=swinT_base, class_num=class_num)

    ### creating a TIA_ViT model for training
    tiavit_net = TIA_ViT(base_model=swinT_base)

    ### creating a TIA_ViT model for visulization
    #tiavit_net = TIA_ViT_Visulization(base_model=swinT_base)

    ### creating a TIA_ViT model for ablation
    #tiavit_net = TIA_ViT_Ablation(base_model=swinT_base)

    with torch.no_grad():
        print('########################## SwinT_summary #########################')
        summary(tiavit_net, torch.randn((1, 3, 224, 224)))
        print('\n', '########################## SwinT_net #########################')
        print(tiavit_net, '\n')

    tiavit_net = tiavit_net.cuda(gpu_device)

    ########################## fitting models and testing models #########################
    #print('########################## fitting models and testing models #########################')
    #Single_out_fit(ddai_net=swinT_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
    #              lr_fn='vit', epoch = epochs, gpu_device = gpu_device,
    #             weight_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx_greece\SwinT.pth')

    Multiple_out_fit(ddai_net=tiavit_net, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                    epoch=epochs, gpu_device=gpu_device,
                     weight_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx_greece\IDANet_3060_r_0.pth')

    ########################## searching the best learning rate #########################
    #print('########################## searching the best learning rate #########################')
    #searching_best_lr(min_lr=1e-7, max_lr=1e-5, fitting_model='single_out',
    #                  train_loader=train_loader, val_loader=val_loader, class_num=class_num,
    #                  test_loader=test_loader, lr_fn='searching_best_lr', epoch=epochs, gpu_device=0,
    #                  onecycle_mr=1e-5)

    ########################## testing function #########################
    '''
    print('########################## testing function #########################')
    larynx_weight = torch.load(
        r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx_greece\IDANet_0-Larynx-greece-2080ti-r0.pth',
    map_location='cuda:0')
    swinT_net.load_state_dict(larynx_weight, strict=True)
    testing_funnction(test_model = swinT_net, train_loader=train_loader, val_loader=val_loader,
                      test_loader=test_loader, gpu_device=gpu_device, out_mode = 'triplet')
    '''


    ########################## interpretable learning section #########################
    #print('########################## interpretable learning section #########################')
    #### ads-model
    #swinT_ads_cam_base = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=ads_cam_cls_num,
    #                                     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
    #                                     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
    #                                     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #                                     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #                                     use_checkpoint=False, fused_window_process=False)
    #swinT_ads_cam_net = SwinT_Net(base_model=swinT_ads_cam_base, class_num=ads_cam_cls_num)

    #ads_weight = torch.load(r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\SwinT_ADS_CAM_Natural_mixed_ab.pth')
    #swinT_ads_cam_net.load_state_dict(ads_weight, strict=True)
    #swinT_ads_cam_net = swinT_ads_cam_net.cuda(gpu_device)

    larynx_weight = torch.load(
        r'E:\SOTA_Model_Interpretable_Learning\SIL_Weights\Larynx_greece\SwinT_0-Larynx-greece-2080ti-r0.pth')
    swinT_net.load_state_dict(larynx_weight, strict=True)
    testing_funnction(test_model=swinT_net, train_loader=test_loader, val_loader=val_loader,
                                        test_loader=test_loader, gpu_device=gpu_device, out_mode='single')

    print(isinstance(swinT_net.layers_0.blocks[0].mlp.fc2, nn.Linear))
    print(swinT_net.norm)
    print(swinT_net.avgp)
    path_name = r'E:\SOTA_Model_Interpretable_Learning\SIL_Datasets\Larynx_greece\Test\III'
    img_name = r'455.jpg'
    img_name_list = os.listdir(path_name)
    image_path = path_name + img_name
    training_img_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\ST_MSLNet\\1-All'
    result_img_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_greece\SwinT_XGrad_CAM\Images\III'
    result_cam_path = r'E:\SOTA_Model_Interpretable_Learning\SIL_Results\Larynx_greece\SwinT_XGrad_CAM\CAMs\III'

    for i in img_name_list:
        #my_ads_cam = Adaptive_Dual_Semantic_CAM(model = swinT_net, path_name = path_name, img_name = i,
        #                                        select_cls = 2, inverse_set = True, set_rate_1 = 0.65,
        #                                        show_all_fp = True, set_rate_2 = 0.65, rd_setting = True,
        #                                        transform=transform, training_img_path = training_img_path,
        #                                        ads_cam_model = None, result_img_path = result_img_path,
        #                                        result_cam_path = result_cam_path, out_mode= 'triplet',
        #                                        adaptive_strategy = 'histopathology')

        #my_cam = CAM(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
        #          select_cls = 0, inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
        #         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

        #my_grad_cam = Grad_CAM(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
        #        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
        #         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

        #my_grad_cam_pp = Grad_CAM_Plus2(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
        #        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
        #         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

        #my_score_cam = Score_CAM(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
        #        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
        #         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

        #my_s_grad_cam_pp = Smooth_Grad_CAM_Plus2(model = swinT_net, path_name = path_name, img_name = i, result_cam_path = result_cam_path,
        #        inverse_set = True, transform = transform, gpu_device = gpu_device, show_all_fp = True,
        #         training_img_path = None,  result_img_path = result_img_path, map_size = [7, 7])

        #my_layer_cam = Layer_CAM(model=swinT_net, path_name=path_name, img_name=i,
        #                result_cam_path=result_cam_path,inverse_set=True, transform=transform, gpu_device=gpu_device,
        #                show_all_fp=True, training_img_path=None, result_img_path=result_img_path, map_size=[7, 7])

        my_xgrad_cam = XGrad_CAM(model=swinT_net, path_name=path_name, img_name=i, result_cam_path=result_cam_path,
                                 inverse_set=True, transform=transform, gpu_device=gpu_device,show_all_fp=True,
                                 training_img_path=None, result_img_path=result_img_path, map_size=[7, 7])

        my_xgrad_cam.get_all_result_images()
