#https://github.com/rwightman/efficientdet-pytorch
#!cd ./efficientdet-pytorch && python setup.py install
import effdet
print(effdet)

import warnings
warnings.filterwarnings("ignore")

import sys
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from ensemble_boxes import weighted_boxes_fusion
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data.dataloader import default_collate

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet import create_model, unwrap_bench, create_loader, create_dataset, create_evaluator, create_model_from_config
from effdet.data import resolve_input_config, SkipSubset
from effdet.anchors import Anchors, AnchorLabeler
from timm.models import resume_checkpoint, load_checkpoint
from timm.utils import *
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

import random

SEED = 1004

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

from pycocotools.coco import COCO
coco = COCO('./dataset/train.json')

train_df = pd.DataFrame()

image_ids = []
class_name = []
class_id = []
area = []
x_min = []
y_min = []
x_max = []
y_max = []
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
for image_id in coco.getImgIds():
        
    image_info = coco.loadImgs(image_id)[0]
    ann_ids = coco.getAnnIds(imgIds=image_info['id'])
    anns = coco.loadAnns(ann_ids)
        
    file_name = image_info['file_name']
        
    for ann in anns:
        image_ids.append(file_name)
        class_name.append(classes[ann['category_id']])
        class_id.append(ann['category_id'])
        area.append(ann['area'])
        x_min.append(float(ann['bbox'][0]))
        y_min.append(float(ann['bbox'][1]))
        x_max.append(float(ann['bbox'][0]) + float(ann['bbox'][2]))
        y_max.append(float(ann['bbox'][1]) + float(ann['bbox'][3]))

train_df['image_id'] = image_ids
train_df['class_name'] = class_name
train_df['class_id'] = class_id
train_df['x_min'] = x_min
train_df['y_min'] = y_min
train_df['x_max'] = x_max
train_df['y_max'] = y_max
train_df['area'] = area

train_df_s = train_df[train_df['area'] < 30000]
train_df_b = train_df[train_df['area'] >= 30000]

df_annotations_wbf = train_df #train_df_b.reset_index(drop=True) #train_df

print(train_df.index, df_annotations_wbf.index)
# print(df_annotations_wbf.image_id.value_counts())

IMG_SIZE = 1024
# IMG_SIZE = 896

class TrainGlobalConfig:
    def __init__(self):
        self.num_classes = 10
        self.num_workers = 2
        # self.batch_size = 16
        self.batch_size = 4 #7
        # self.batch_size = 1 #16
        self.n_epochs = 20
        self.lr = 0.0002
        # self.model_name = 'tf_efficientdet_d1'
        self.model_name = 'tf_efficientdet_d4_ap'
        # self.model_name = 'tf_efficientdet_d3_ap'
        self.folder = 'training_job'
        self.verbose = True
        self.verbose_step = 1
        self.step_scheduler = True
        self.validation_scheduler = False
        self.n_img_count = len(df_annotations_wbf.image_id.unique())
        # self.n_img_count = len(train_df.image_id.unique())
        self.OptimizerClass = torch.optim.AdamW
        self.SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        self.scheduler_params = dict(
                            T_0=50,
                            T_mult=1,
                            eta_min=0.0001,
                            last_epoch=-1,
                            verbose=False
                            )
        self.kfold = 5
    
    def reset(self):
        self.OptimizerClass = torch.optim.AdamW
        self.SchedulerClass = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts

train_config = TrainGlobalConfig()


from sklearn.model_selection import GroupKFold, train_test_split, StratifiedKFold

df_annotations_wbf['fold'] = -1
# group_kfold  = GroupKFold(n_splits = 3)
# for fold, (train_index, val_index) in enumerate(group_kfold.split(df_annotations_wbf, groups=df_annotations_wbf.image_id.tolist())):

X = df_annotations_wbf
y = df_annotations_wbf.class_id

kfold  = StratifiedKFold(n_splits = 5)
for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
    df_annotations_wbf.loc[val_index, 'fold'] = fold
df_annotations_wbf.sample(5)



# TRAIN_ROOT_PATH = '../input/vinbigdata-original-image-dataset/vinbigdata/train'
TRAIN_ROOT_PATH = './dataset/'

class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()
        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test
        
    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        image, boxes, labels = self.load_image_and_boxes(index)
        # print(type(image), type(boxes), type(labels), random.random())
        
        if self.test or random.random() > 0.5:
            image, boxes, labels = self.load_image_and_boxes(index)
        else:
            image, boxes, labels = self.load_cutmix_image_and_boxes(index)

        # if self.test or random.random() > 0.33:
        #     image, boxes, labels = self.load_image_and_boxes(index)
        # elif random.random() > 0.5:
        #     image, boxes, labels = self.load_cutmix_image_and_boxes(index)
        # else:
        #     image, boxes, labels = self.load_mixup_image_and_boxes(index)
        
        ## To prevent ValueError: y_max is less than or equal to y_min for bbox from albumentations bbox_utils
        labels = np.array(labels, dtype=np.int).reshape(len(labels), 1)
        combined = np.hstack((boxes.astype(np.int), labels))
        combined = combined[np.logical_and(combined[:,2] > combined[:,0],
                                                          combined[:,3] > combined[:,1])]
        boxes = combined[:, :4]
        labels = combined[:, 4].tolist()
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels)
        target['image_id'] = torch.tensor([index])
        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]  ## ymin, xmin, ymax, xmax
                    break
            
            ## Handling case where no valid bboxes are present
            if len(target['boxes'])==0 or i==9:
                return None
            else:
                ## Handling case where augmentation and tensor conversion yields no valid annotations
                try:
                    assert torch.is_tensor(image), f"Invalid image type:{type(image)}"
                    assert torch.is_tensor(target['boxes']), f"Invalid target type:{type(target['boxes'])}"
                except Exception as E:
                    print("Image skipped:", E)
                    return None      

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
#         print(f'{TRAIN_ROOT_PATH}/{image_id}.jpg')
        # image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR).copy()
        image = cv2.imread(f'{TRAIN_ROOT_PATH}{image_id}', cv2.IMREAD_COLOR).copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image_id'] == image_id]
        boxes = records[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = records['class_id'].tolist()
        resize_transform = A.Compose(
                                    [A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0)], 
                                    # [A.Resize(height=512, width=512, p=1.0)], 
                                    p=1.0, 
                                    bbox_params=A.BboxParams(
                                        format='pascal_voc',
                                        min_area=0.1, 
                                        min_visibility=0.1,
                                        label_fields=['labels'])
                                    )

        resized = resize_transform(**{
                'image': image,
                'bboxes': boxes,
                'labels': labels
            })

        resized_bboxes = np.vstack((list(bx) for bx in resized['bboxes']))
        return resized['image'], resized_bboxes, resized['labels']
    
    def load_mixup_image_and_boxes(self, index):
        image, boxes, labels = self.load_image_and_boxes(index)
        r_image, r_boxes, r_labels = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        return (image+r_image)/2, np.vstack((boxes, r_boxes)).astype(np.int32), np.concatenate((labels, r_labels))

    # def load_cutmix_image_and_boxes(self, index, imsize=512):
    def load_cutmix_image_and_boxes(self, index, imsize=IMG_SIZE):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = np.array([], dtype=np.int)

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels = np.concatenate((result_labels, labels))

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        index_to_use = np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)
        result_boxes = result_boxes[index_to_use]
        result_labels = result_labels[index_to_use]
        
        return result_image, result_boxes, result_labels
    
# train_dataset = DatasetRetriever(image_ids=train_ids, marking=df_annotations_wbf, transforms=get_train_transforms(), test=False)
# next(iter(train_dataset))

def get_train_transforms():
    return A.Compose(
        [
        ## RandomSizedCrop not working for some reason. I'll post a thread for this issue soon.
        ## Any help or suggestions are appreciated.
#         A.RandomSizedCrop(min_max_height=(300, 512), height=512, width=512, p=0.5),
#         A.RandomSizedCrop(min_max_height=(300, 1000), height=1000, width=1000, p=0.5),
        # A.OneOf([
        #     A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
        #                          val_shift_limit=0.2, p=0.9),
        #     A.RandomBrightnessContrast(brightness_limit=0.2, 
        #                                contrast_limit=0.2, p=0.9),
        # ],p=0.9),
        # A.JpegCompression(quality_lower=85, quality_upper=95, p=0.2),
        # A.OneOf([
        #     A.Blur(blur_limit=3, p=1.0),
        #     A.MedianBlur(blur_limit=3, p=1.0)
        #     ],p=0.1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # A.Transpose(p=0.5),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        # A.Resize(height=512, width=512, p=1),
        # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
        ToTensorV2(p=1.0)
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            # A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

class Fitter:
    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5
        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = config.OptimizerClass(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader, fold):
        history_dict = {}
        history_dict['epoch'] = []
        history_dict['train_loss'] = []
        history_dict['val_loss'] = []
        history_dict['train_lr'] = []
        
        for e in range(self.config.n_epochs):
            history_dict['epoch'].append(self.epoch)
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.utcnow().isoformat()
            
            if self.config.verbose:
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss, loss_trend, lr_trend = self.train_epoch(train_loader)
            history_dict['train_loss'].append(loss_trend)
            history_dict['train_lr'].append(lr_trend)
            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/{fold}-{IMG_SIZE}-last-checkpoint.bin')
            
            t = time.time()
            summary_loss, loss_trend = self.validation(validation_loader)
            history_dict['val_loss'].append(loss_trend)
            self.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/{fold}-{IMG_SIZE}-best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
                
                try:
                    os.remove(f)
                except:pass
                f = f'{self.base_dir}/{fold}-{IMG_SIZE}-best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin'

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1
        return history_dict

    def train_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        loss_trend = []
        lr_trend = []
        for step, (images, targets, image_ids) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )            
            
            images = torch.stack(images)
            images = images.to(self.device).float()
            
            target_res = {}
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]
            target_res['bbox'] = boxes
            target_res['cls'] = labels
            self.optimizer.zero_grad()
            output = self.model(images, target_res)

            loss = output['loss']
            loss.backward()
            summary_loss.update(loss.detach().item(), self.config.batch_size)
            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

            
            lr = self.optimizer.param_groups[0]['lr']
            loss_trend.append(summary_loss.avg)
            lr_trend.append(lr)
        return summary_loss, loss_trend, lr_trend
    
    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        loss_trend = []
#         lr_trend = []
        
        for step, (images, targets, image_ids) in tqdm(enumerate(val_loader), total=len(val_loader)):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                images = images.to(self.device).float()
                target_res = {}
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]
                target_res['bbox'] = boxes
                target_res['cls'] = labels 
                target_res["img_scale"] = torch.tensor([1.0] * self.config.batch_size,
                                                       dtype=torch.float).to(self.device)
                target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * self.config.batch_size,
                                                      dtype=torch.float).to(self.device)
                
                output = self.model(images, target_res)
            
                loss = output['loss']
                summary_loss.update(loss.detach().item(), self.config.batch_size)

                loss_trend.append(summary_loss.avg)
        return summary_loss, loss_trend[-1]
    
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
        print('epoch:', self.epoch, 'best_summary_loss:',self.best_summary_loss)
        return self.model
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
            
## Training will resume if the checkpoint path is specified below
checkpoint_path = f'2-{IMG_SIZE}-best-checkpoint-040epoch.bin'
# checkpoint_path = None #'last-checkpoint.bin'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

## Filters out invalid return items from the Dataloader
# def collate_fn(batch):
#     batch = list(filter(lambda x : x is not None, batch))
#     return default_collate(batch)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    return tuple(zip(*batch))

# def collate_fn(batch):
#     return tuple(zip(*batch))

fold_history = []
for val_fold in range(train_config.kfold):
    print(f'Fold {val_fold+1}/{train_config.kfold}')
    
    train_ids = df_annotations_wbf[df_annotations_wbf['fold'] != val_fold].image_id.unique()
    val_ids = df_annotations_wbf[df_annotations_wbf['fold'] == val_fold].image_id.unique()
    
    train_dataset = DatasetRetriever(
                        image_ids=train_ids,
                        marking=df_annotations_wbf,
                        transforms=get_train_transforms(),
                        test=False,
                        )

    validation_dataset = DatasetRetriever(
                            image_ids=val_ids,
                            marking=df_annotations_wbf,
                            transforms=get_valid_transforms(),
                            test=True,
                            )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )
    
    base_config = get_efficientdet_config(train_config.model_name)
    base_config.image_size = (IMG_SIZE, IMG_SIZE) #(512, 512)

    if(checkpoint_path):
        print(f'Resuming from checkpoint: {checkpoint_path}')        
        model = create_model_from_config(base_config, bench_task='train', bench_labeler=True,
                                 num_classes=train_config.num_classes,
                                 pretrained=False)
        model.to(device)
        
        fitter = Fitter(model=model, device=device, config=train_config)
        # fitter.load(f"./training_job/{val_fold+1}-{IMG_SIZE}-{checkpoint_path}")
        # fitter.load(f"./training_job/1-{IMG_SIZE}-{checkpoint_path}")
        fitter.load(f"./training_job/{checkpoint_path}")
    
    else:
        model = create_model_from_config(base_config, bench_task='train', bench_labeler=True,
                                     pretrained=True,
                                     num_classes=train_config.num_classes)
        model.to(device)
    
        fitter = Fitter(model=model, device=device, config=train_config)  
        
    model_config = model.config
    history_dict = fitter.fit(train_loader, val_loader, val_fold+1)
    fold_history.append(history_dict)
    
    ## Reset Optimizer and LR Sch+)
