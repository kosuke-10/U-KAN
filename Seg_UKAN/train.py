import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml

from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs

import losses
from dataset import Dataset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8') 

ARCH_NAMES = ['UKAN']
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')



def plot_progress_realtime(log, output_dir, exp_name, epoch, best_iou, best_dice):
    """リアルタイムで学習進行を可視化"""
    if len(log['epoch']) == 0:
        return
        
    # 5エポックごとまたは最初の10エポック、または最良モデル保存時に生成
    if epoch % 5 == 0 or epoch < 10 or len(log['epoch']) == 1:
        df = pd.DataFrame(log)
        
        plt.figure(figsize=(16, 10))
        
        # 2x3のサブプロット作成
        plt.subplot(2, 3, 1)
        plt.plot(df['epoch'], df['loss'], 'b-', linewidth=2, label='Training Loss')
        plt.plot(df['epoch'], df['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        plt.plot(df['epoch'], df['iou'], 'b-', linewidth=2, label='Training IoU')
        plt.plot(df['epoch'], df['val_iou'], 'r-', linewidth=2, label='Validation IoU')
        
        # 最良IoUにマーカー追加
        if len(df) > 0:
            best_epoch_idx = df['val_iou'].idxmax()
            best_epoch_num = df.loc[best_epoch_idx, 'epoch']
            best_iou_val = df.loc[best_epoch_idx, 'val_iou']
            plt.scatter([best_epoch_num], [best_iou_val], color='gold', s=100, marker='*', 
                       label=f'Best: {best_iou_val:.4f}', zorder=5)
        
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.title(f'IoU Curves (Current: {df["val_iou"].iloc[-1]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(df['epoch'], df['val_dice'], 'g-', linewidth=2, label='Validation Dice')
        
        # 5エポック以上ある場合は移動平均も表示
        if len(df) >= 5:
            smooth_dice = df['val_dice'].rolling(window=5, min_periods=1).mean()
            plt.plot(df['epoch'], smooth_dice, 'darkgreen', linewidth=3, alpha=0.8, 
                    label='5-epoch Moving Avg')
        
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title(f'Dice Score (Current: {df["val_dice"].iloc[-1]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 4)
        plt.plot(df['epoch'], df['lr'], 'purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 学習進行状況
        plt.subplot(2, 3, 5)
        progress_pct = (epoch + 1) / 400 * 100
        remaining_epochs = 400 - (epoch + 1)
        
        # プログレスバー風の表示
        completed = int(progress_pct / 2)  # 50文字のバーにスケール
        remaining = 50 - completed
        progress_bar = '█' * completed + '░' * remaining
        
        plt.text(0.1, 0.8, f'Progress: {progress_pct:.1f}%', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.6, f'Epoch: {epoch + 1}/400', fontsize=12)
        plt.text(0.1, 0.4, f'Remaining: {remaining_epochs} epochs', fontsize=12)
        plt.text(0.1, 0.2, progress_bar, fontsize=8, fontfamily='monospace')
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Training Progress')
        
        # 統計サマリー
        plt.subplot(2, 3, 6)
        current_loss = df['val_loss'].iloc[-1]
        current_iou = df['val_iou'].iloc[-1]
        current_dice = df['val_dice'].iloc[-1]
        
        # README.mdの目標値との比較
        target_iou = 0.6526
        target_dice = 0.7875
        
        iou_progress = (current_iou / target_iou) * 100 if target_iou > 0 else 0
        dice_progress = (current_dice / target_dice) * 100 if target_dice > 0 else 0
        
        stats_text = f"""Current Performance:
        
IoU: {current_iou:.4f} (Target: {target_iou:.4f})
Progress: {iou_progress:.1f}%

Dice: {current_dice:.4f} (Target: {target_dice:.4f})
Progress: {dice_progress:.1f}%

Best IoU: {best_iou:.4f}
Best Dice: {best_dice:.4f}

Val Loss: {current_loss:.4f}"""
        
        plt.text(0.05, 0.95, stats_text, fontsize=10, fontfamily='monospace',
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightblue", alpha=0.8))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Performance Summary')
        
        # 全体のタイトル
        plt.suptitle(f'🚀 U-KAN Training Progress: {exp_name} (Epoch {epoch + 1})', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{exp_name}/progress.png', dpi=200, bbox_inches='tight')
        plt.close()  # メモリリークを防ぐ
        
        # 簡潔なログ出力
        if epoch % 10 == 0 or epoch < 5:
            print(f"📊 Progress visualization updated: {output_dir}/{exp_name}/progress.png")


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--no_kan', action='store_true')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=config['no_kan'])

    model = model.cuda()


    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  
    

    
    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    # 340行目付近を以下のように修正:
    train_transform = Compose([
        RandomRotate90(),
        # geometric.transforms.Flip(),  # 古いAPI
        transforms.Flip(),  # 新しいAPI
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])


    best_iou = 0
    best_dice= 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        # 🎨 リアルタイム可視化を追加（ここに挿入！）
        plot_progress_realtime(log, output_dir, exp_name, epoch, best_iou, best_dice)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/best_model.pth')  # best.pthに変更
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print(f"=> saved best model (epoch {epoch}, IoU: {best_iou:.4f})")
            trigger = 0
            
            # 🏆 最良モデル保存時は必ず可視化更新
            plot_progress_realtime(log, output_dir, exp_name, epoch, best_iou, best_dice)

        # 最終エポックでの保存を追加
        if epoch == config['epochs'] - 1:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/last_model.pth')  # last_model.pthに変更
            print("=> saved last model")

            # 🏁 最終エポックでも可視化更新
            plot_progress_realtime(log, output_dir, exp_name, epoch, best_iou, best_dice)

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            # Early stopping時も可視化更新
            plot_progress_realtime(log, output_dir, exp_name, epoch, best_iou, best_dice)
            break

        torch.cuda.empty_cache()
            
if __name__ == '__main__':
    main()
