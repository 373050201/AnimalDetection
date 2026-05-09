import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import torchvision.transforms.functional as F
import random



class YoloDataset(Dataset):
    def __init__(self,img_folder,label_folder,img_transform,label_transform,aug=False):
        self.img_folder=img_folder
        self.label_folder=label_folder
        self.img_transform=img_transform
        self.label_transform=label_transform
        self.img_names=os.listdir(self.img_folder)
        self.aug=aug
    def __len__(self):
        return len(self.img_names)
    def __getitem__(self, idx):
        img_name=self.img_names[idx]
        img_path=os.path.join(self.img_folder,img_name)
        img=Image.open(img_path)
        label_name=img_name.split('.')[0]+".txt"
        label_path=os.path.join(self.label_folder,label_name)
        target=[]
        with open(label_path) as f:
            label_content=f.read()
            obj_infos=label_content.strip().split("\n")
            for obj_info in obj_infos:
                info_list=obj_info.strip().split(" ")
                class_id = int(info_list[0])
                center_x = float(info_list[1])
                center_y = float(info_list[2])
                width = float(info_list[3])
                height = float(info_list[4])
                if not self.aug:#无数据增强
                    target=[class_id, center_x, center_y, width, height]
                else:#数据增强
                    if random.random()<0.3:#30%概率调整亮度
                        brightness_factor=random.uniform(0.8,1.2)#亮度范围
                        img=F.adjust_brightness(img,brightness_factor)
                    if random.random()<0.3:#30%概率调整对比度
                        contrast_factor=random.uniform(0.8,1.2)
                        img=F.adjust_contrast(img,contrast_factor)
                    if random.random()<0.3:##30%概率调整饱和度
                        saturation_factor=random.uniform(0.8,1.2)
                        img=F.adjust_saturation(img,saturation_factor)
                    target=[class_id, center_x, center_y, width, height]
                    if random.random()<0.5:#50%概率水平翻转
                        img=F.hflip(img)
                        target[1]=1.0-target[1]#翻转center_x
        if self.img_transform is not None:
            img=self.img_transform(img)
        if self.label_transform is not None:
            target=self.label_transform(target)
        return img,target



def format_convert(x):
    """
    input:(list)[cls_id,x_center,y_center,width,height]
    output:(tensor)[x_center,y_center,width,height,cls1,cls2,cls3]
    """
    x=torch.tensor(x)
    if x[0].item()==0:
        x=torch.cat((x,torch.tensor([1.0,0.0,0.0])),0)
    elif x[0].item()==1:
        x=torch.cat((x,torch.tensor([0.0,1.0,0.0])),0)
    elif x[0].item()==2:
        x=torch.cat((x,torch.tensor([0.0,0.0,1.0])),0)
    x=x[1:]
    return x



if __name__=="__main__":
    train_dataset=YoloDataset("datasets/animals_yolo/images/train",
                              "datasets/animals_yolo/labels/train",
                              img_transform=transforms.Compose([
                                    transforms.Resize((640, 640)),
                                    transforms.ToTensor()
                                ]),
                              label_transform=format_convert)
    img,target=train_dataset[0]
    #print(img)
    target.unsqueeze_(0)
    print(target)
    print(target.shape)
    print(len(train_dataset))