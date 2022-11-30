import torch
from torchvision.models import detection
from torchvision import datasets, models, transforms
from PIL import Image
import cv2

model = detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True)
model.eval()

# check_point = './save/faster_rcnn.pth' # 체크포인트 경로    
# model.load_state_dict(torch.load(check_point))
# model.eval()    

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([
    ## Resize는 사용하지 않고 원본을 추출
   # transforms.Resize((224,224)),
   transforms.ToTensor(),
   # normalize
])
# PIL_image = Image.fromarray(img)
# img_input = preprocess(PIL_image)
# detections = model(img_input.unsqueeze(0))[0]

# print(detections)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    return tuple(zip(*batch))

from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO

class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.predictions = {
            "images": self.coco.dataset["images"].copy(),
            "categories": self.coco.dataset["categories"].copy(),
            "annotations": None
        }
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)
        boxes = np.array([x['bbox'] for x in anns])

        # boxex (x_min, y_min, x_max, y_max)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        
        # torchvision faster_rcnn은 label=0을 background로 취급
        # class_id를 1~10으로 수정 
        labels = np.array([x['category_id']+1 for x in anns]) 
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = np.array([x['area'] for x in anns])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        is_crowds = np.array([x['iscrowd'] for x in anns])
        is_crowds = torch.as_tensor(is_crowds, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([index]), 'area': areas,
                  'iscrowd': is_crowds}

        # transform
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)

        return image, target, image_id
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())
    
def get_train_transform():
    return A.Compose([
        A.Resize(1024, 1024),
        # A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),        
        # A.VerticalFlip(p=0.5),              
        # A.RandomRotate90(p=0.5),        
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        A.Flip(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


annotation = './dataset/train.json' # annotation 경로
data_dir = './dataset' # data_dir 경로
train_dataset = CustomDataset(annotation, data_dir, get_train_transform()) 
train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    # batch_size=22,
    # shuffle=True,
    # num_workers=2,
    collate_fn=collate_fn
)


from tqdm import tqdm
import os
import numpy as np

def train_fn(num_epochs, train_data_loader, optimizer, model, device):
    best_loss = 1000
    loss_hist = Averager()
    for epoch in range(num_epochs):
        loss_hist.reset()

        for images, targets, image_ids in tqdm(train_data_loader):

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = model(images, targets)
            # print(type(loss_dict), len(loss_dict))
            # print(loss_dict)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            # backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch #{epoch+1} loss: {loss_hist.value}")
        if loss_hist.value < best_loss:
            save_path = './save/faster_rcnn.pth'
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            torch.save(model.state_dict(), save_path)
            best_loss = loss_hist.value
            print('file saved', loss_hist.value, '(best:', best_loss,')')            
            
import torch
import torchvision
# from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 11 # class 개수= 10 + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#백본 변경
# from torchvision.models.detection.rpn import AnchorGenerator
# # backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# backbone.out_channels = 1280
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
# model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
# model.eval()

# def get_model_instance_segmentation(num_classes):
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
#     return model

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]

# optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = 1000
optimizer = torch.optim.AdamW(params, lr=0.0005)
# num_epochs = 50

#check_point load 추가
check_point = './save/faster_rcnn.pth' # 체크포인트 경로    
model.load_state_dict(torch.load(check_point))
# model.eval()    

# training
train_fn(num_epochs, train_data_loader, optimizer, model, device)