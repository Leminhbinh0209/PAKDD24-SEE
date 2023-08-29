
import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from pathlib import Path
from functools import partial
import PIL
import multiprocessing
from losses import *

from sampler import UniqueClassSempler, UniqueClassSampler, BalancedSampler
from helpers import get_emb, evaluate
from dataset import CUBirds, SOP, Cars
from dataset.Inshop import Inshop_Dataset
from models.model import init_model
        
def get_args_parser():
    parser = argparse.ArgumentParser('SEE', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str,
        choices=['resnet50', 
                 'deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'dino_vits16'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--image_size', type=int, default=224, help="""Size of Global Image""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--emb', default=128, type=int, help="""Dimensionality of output for [CLS] token.""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--clip_grad', type=float, default=0.1, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=90, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=1e-5, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--fc_lr_scale", default=1, type=float)
    parser.add_argument("--warmup_epochs", default=1, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr_scale', type=float, default=0.1, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['sgd', 'adam', 'adamw',  'adamp', 'radam'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--pool', default = 'token', type=str, choices=['token', 'avg'], help = 'ViT Pooling')
    parser.add_argument('--lr_decay', default = None, type=str, help = 'Learning decay step setting')
    parser.add_argument('--lr_decay_gamma', default = None, type=float, help = 'Learning decay step setting')
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--bn_freeze', type=bool, default=True)
    parser.add_argument('--use_lastnorm', type=bool, default=True)

    # Augementation parameters
    parser.add_argument('--global_crops_number', type=int, default=1, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    # Spherical embedding expansion 

    parser.add_argument('--n-expan', default = 16, type = int,  help = 'Number spherical expansion'  )
    parser.add_argument('--lower-bound', default = 0.0, type = float,  help = 'Positive threshold to expand' )
    parser.add_argument('--use-schedule', default = False, type = lambda x: (str(x).lower() == 'true'), help = 'Use schedule to calculate positive threshold')
    parser.add_argument('--max-thresh', default = 0.4, type = float, help = 'Maximum Positive threshold to expand' )
    parser.add_argument('--min-thresh', default = 0.04, type = float,help = 'Minimum Positive threshold to expand' )
    parser.add_argument('--type', default = 1, type = int, help = 'Type of schedule for updating threshold' )
    parser.add_argument('--lambd', default = 0.2, type = float, help = 'Weight loss for aug samples')
    parser.add_argument('--random', type=bool, default=False)

    # Hyperbolic MetricLearning parameters
    parser.add_argument('--IPC', type=int, default=2)
    parser.add_argument('--clip_r', type=float, default=2.3)
    parser.add_argument('--save_emb', type=utils.bool_flag, default=False)
    parser.add_argument('--best_recall', nargs="+",  type=int, default=[0])
    parser.add_argument('--loss', default='PA', type=str, choices=['PA', 'MS', 'PNCA', 'SoftTriple', 'SupCon', 'seeproxy'])
    parser.add_argument('--cluster_start', default=0, type=int)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--num_hproxies', default=512, type=int, help="""Dimensionality of output for [CLS] token.""")

    parser.add_argument('--mrg', type=float, default=0.1)
    

    parser.add_argument('--augmentation', default=None, type=str, 
                        choices=["cutmix", "mixup", None], help="Apply data augmenetaion for training .")
    


    # Misc
    parser.add_argument('--dataset', default='CUB', type=str, 
                        choices=["SOP", "CUB", "Cars", "Inshop"], help='Please specify dataset to train')
    parser.add_argument('--data_path', default='/home/data/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="logs/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--run_name', default="", type=str, help='Wandb run name')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--eval_freq', default=1, type=int, help='Evaluation for every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, dest = 'local_rank', help="Please ignore and do not set this argument.")

    return parser

def train_one_epoch(model,  sup_metric_loss, get_emb_s, data_loader, optimizer, 
                    lr_schedule, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    if args.use_schedule and (args.loss in ['seeproxy']):        
            sup_metric_loss.update_threshold(epoch, args.epochs)
            callback = dict()
            callback["rate"] = 0.0
            callback["pos_cos"] = []
            callback["batch_loss"] = 0.0
            callback["aug_loss"] = 0.0
            n_origs = 0
            n_expans = 0
    model.train()

    for it, (x, y, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):        
        it = len(data_loader) * epoch + it  
        for i, param_group in enumerate(optimizer.param_groups):
            lr = args.lr * (args.batch_size * utils.get_world_size()) / 180.
            param_group["lr"] = lr * param_group["lr_scale"] * lr_schedule[it]
            
            if epoch < args.warmup_epochs and "pretrained_params" in param_group["name"]:
                param_group["lr"] = 0
            elif "pretrained_params" in param_group["name"]:
                param_group["lr"] = lr * param_group["lr_scale"] * lr_schedule[it]
        
        x = torch.cat([im.cuda(non_blocking=True) for im in x])
        y = y.cuda(non_blocking=True).repeat(args.global_crops_number)


        if utils.is_main_process():
            r = np.random.rand(1)
            r_tensor = torch.tensor(r).cuda(non_blocking=True)  # Convert to a tensor and move to GPU
        else:
            r_tensor = torch.zeros(1).cuda(non_blocking=True)  # Create a dummy tensor on other processe
        # Broadcast  the value of lam_tensor from rank 0 to all other processes
        dist.broadcast(r_tensor, src=0)
        r = r_tensor.item()  # Convert back to a Python scalar
        augmentation = args.augmentation if r < 0.5 else None


        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if augmentation is None:
                z = model(x) 
                if args.loss == 'SupCon' and args.IPC > 0:
                    z = z.view(len(z) // args.IPC, args.IPC, args.emb)
                if world_size > 1:
                    z = utils.all_gather(z, args.local_rank)
                    y = utils.all_gather(y, args.local_rank)
                    
                if args.loss in ['seeproxy']:
                    loss1 = sup_metric_loss(z, y, callback) 
                    n_origs += len(y)
                    n_expans += callback["rate"]
                else:
                    loss1 = sup_metric_loss(z, y) 



                
                loss = loss1

            elif  augmentation == 'cutmix':
                """
                Only  PA is implement here
                """
                assert args.loss == "PA", f"Cannot apply other loss function in data augmentation ablation studies"
                # Generate lam on the master process (rank 0)
                if utils.is_main_process():
                    lam = np.random.beta(1.0, 1.0)
                    lam_tensor = torch.tensor(lam).cuda(non_blocking=True)  # Convert to a tensor and move to GPU
                else:
                    lam_tensor = torch.zeros(1).cuda(non_blocking=True)  # Create a dummy tensor on other processes
                # Broadcast the value of lam_tensor from rank 0 to all other processes
                dist.broadcast(lam_tensor, src=0)
                lam = lam_tensor.item()  # Convert back to a Python scalar


                rand_index = torch.randperm(x.size()[0]).cuda()
                target_a = y
                target_b = y[rand_index]
                bbx1, bby1, bbx2, bby2 = utils.rand_bbox(x.size(), lam)
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                z = model(x) 
                if world_size > 1:
                    target_a = utils.all_gather(target_a, args.local_rank)
                    target_b = utils.all_gather(target_b, args.local_rank)
                    z = utils.all_gather(z, args.local_rank)
                loss1 = loss = sup_metric_loss(z,target_a) *  lam + sup_metric_loss(z,target_b) *  (1.0-lam)

            elif augmentation == 'mixup':
                assert args.loss == "PA", f"Cannot apply other loss function in data augmentation ablation studies"

                if utils.is_main_process():
                    lam = np.random.beta(1.0, 1.0)
                    lam_tensor = torch.tensor(lam).cuda(non_blocking=True)  # Convert to a tensor and move to GPU
                else:
                    lam_tensor = torch.zeros(1).cuda(non_blocking=True)  # Create a dummy tensor on other processes
                # Broadcast the value of lam_tensor from rank 0 to all other processes
                dist.broadcast(lam_tensor, src=0)
                lam = lam_tensor.item()  # Convert back to a Python scalar



                rand_index = torch.randperm(x.size()[0]).cuda()
                target_a = y
                target_b = y[rand_index]
                x = x*lam +  x[rand_index]*(1.0-lam)
                z = model(x) 

                if world_size > 1:
                    target_a = utils.all_gather(target_a, args.local_rank)
                    target_b = utils.all_gather(target_b, args.local_rank)
                    z = utils.all_gather(z, args.local_rank)
                loss1 = loss = sup_metric_loss(z,target_a) *  lam + sup_metric_loss(z,target_b) *  (1.0-lam)

        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(False):
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad > 0:
                    param_norms = utils.clip_gradients_value(model, 10, losses=[ sup_metric_loss])
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients_value(model, 10, losses=[ sup_metric_loss])
                    
                    
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                
        torch.cuda.synchronize()
        metric_logger.update(metric_loss=loss1.item())

        metric_logger.update(total_loss=loss.item())
        if args.loss in ['seeproxy']:
            metric_logger.update(aug_rate= (n_expans-n_origs)*1.0 / (n_origs))

        
    metric_logger.synchronize_between_processes()
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    rh_model = 0
    if (epoch % args.eval_freq == 0) and (epoch >= args.warmup_epochs):
        if (args.dataset == "CUB" or args.dataset == "Cars") or epoch >= 40:
            recall_at_ks = evaluate(get_emb_s, args.dataset)
            if utils.is_main_process():
                print("Curr results: ", " | ".join([f"R@{u}: {100*v:.2f}" for (u,v) in zip(args.k_list, recall_at_ks)]))
            rh_model = recall_at_ks[0]
            return_dict.update({"R@1_head": rh_model})

    if (epoch % args.eval_freq == 0) and (args.best_recall[0] < rh_model):
        args.best_recall = recall_at_ks
        
    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser('SEE', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
   

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Directory for Log
    if args.loss in ['seeproxy']:
        wandbname = 'ddp_emb{}_bs{}_lr{}_lrstep{}_lrgamma{}_maxmin{}_{}_lambd{}_nexpand{}_bnfree{}_{}'.format(args.emb, args.batch_size, 
                      args.lr,args.lr_decay, args.lr_decay_gamma, args.max_thresh, args.min_thresh, args.lambd, args.n_expan, args.bn_freeze, args.run_name)
        args.output_dir = args.output_dir + '/logs_{}/{}/{}/{}'.format(args.dataset, args.model, args.loss, wandbname)
    else:
        wandbname ='ddp_emb{}_bs{}_lr{}_lrstep{}_lrgamma{}{}'.format( args.emb,  args.batch_size, args.lr, args.lr_decay, args.lr_decay_gamma, args.run_name)
        args.output_dir = args.output_dir + '/logs_{}/{}/{}/{}'.format(args.dataset, args.model, args.loss, wandbname)
    if not os.path.exists('{}'.format(args.output_dir)):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    
    if args.local_rank == 0:
        wandb.init(project="see_metric_{}".format(args.dataset), 
                    dir=args.output_dir,
                   name="{}_{}".format(args.model, wandbname), 
                   config=args)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    world_size = utils.get_world_size()

    if args.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif args.model == "bn_inception":
        mean_std = (104, 117, 128), (1,1,1)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tr = utils.MultiTransforms(mean_std, model=args.model, view=args.global_crops_number, sz_crop=args.crop_size)
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    if args.dataset == "SOP":
        args.k_list = [1, 10, 100, 1000]
    elif args.dataset == "Inshop":
        args.k_list = [1, 10, 20, 30]
    else:
        args.k_list = [1, 2, 4, 8]

    ds_class = ds_list[args.dataset]
    ds_train = ds_class(args.data_path, "train", train_tr)
    nb_classes = len(list(set(ds_train.ys)))
    if args.IPC > 0:
        sampler = UniqueClassSampler(ds_train.ys, args.batch_size, args.IPC, args.local_rank, world_size)
    else:
        sampler = torch.utils.data.DistributedSampler(ds_train, shuffle=True)
    data_loader = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=True,
    )

    model = init_model(args)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=True, find_unused_parameters=(args.model == 'bn_inception'))

    if args.loss == 'MS':
        sup_metric_loss = MSLoss_Angle().cuda()
    elif args.loss == 'PA':
        sup_metric_loss = PALoss_Angle(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss == 'SoftTriple':
        sup_metric_loss = SoftTripleLoss_Angle(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss == 'PNCA':
        sup_metric_loss = PNCALoss_Angle(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss =='SupCon':
        sup_metric_loss = SupCon(IPC=args.IPC).cuda()    
    elif args.loss == 'seeproxy':
        sup_metric_loss = SEEProxyAnchor(nb_classes=nb_classes,
                                    sz_embed=args.emb,
                                    n_expansion=args.n_expan,
                                    keep_grad=False,
                                    lower_bound=args.lower_bound,
                                    max_thresh=args.max_thresh, 
                                    min_thresh=args.min_thresh,
                                    _type=args.type,
                                    _lambda=args.lambd,
                                    random=args.random).cuda()

        
    params_groups = utils.get_params_groups(model, sup_metric_loss, fc_lr_scale=args.fc_lr_scale, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, eps=1e-4 if args.use_fp16 else 1e-8)  # to use with ViTs
    elif args.optimizer == "adamp":
        from adamp import AdamP
        optimizer = AdamP(params_groups)  # to use with ViTs
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params_groups, eps=1e-4 if args.use_fp16 else 1e-8)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, momentum=0.9, lr=args.lr)  # lr is set by scheduler
        
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    if args.lr_decay == 'cosine':
        lr_schedule = utils.cosine_scheduler(1, args.min_lr_scale, args.epochs, len(data_loader))
    elif args.lr_decay is not None:
        lr_schedule = utils.step_scheduler(int(args.lr_decay), args.epochs, len(data_loader), gamma=args.lr_decay_gamma)
    else:
        lr_schedule = utils.step_scheduler(args.epochs, args.epochs, len(data_loader), gamma=1)

    get_emb_s = partial(
        get_emb,
        model=model.module,
        ds=ds_class,
        path=args.data_path,
        mean_std=mean_std,
        world_size=world_size,
        resize=args.resize_size,
        crop=args.crop_size,
    )

    cudnn.benchmark = True
    for epoch in range(args.epochs):
        if sampler is not None and args.IPC > 0:
            sampler.set_epoch(epoch)
        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(model, sup_metric_loss, get_emb_s, data_loader, optimizer, 
                                      lr_schedule, epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.local_rank == 0:
            with (Path("{}/{}_{}_log.txt".format(args.output_dir, args.dataset, args.model))).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with (Path("{}/{}_{}_best_result_log.txt".format(args.output_dir, args.dataset, args.model))).open("w") as f:
                for (u,v) in zip(args.k_list, args.best_recall):
                        f.write("Best Recall@{}: {:.4f}\n".format(u, v * 100))
            wandb.log(train_stats, step=epoch)
            print("Best results: ", " | ".join([f"R@{u}: {100*v:.2f}" for (u,v) in zip(args.k_list, args.best_recall)]))

        

        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
