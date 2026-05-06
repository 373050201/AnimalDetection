import torch
from torch import nn
from yolo_dataset import YoloDataset
import torchvision.transforms as transforms
import torchvision.models as models



img_size=640

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        #==========1、特征提取模块==========
        self.extract=mobilenet.features#shape:1280*(img_size/32)*(img_size/32)=1280*20*20
        #==========2、特征增强模块==========
        self.channel_attention=nn.Sequential(#通道注意力
            nn.AdaptiveAvgPool2d(1),#shape:1280*1*1
            nn.Conv2d(1280,80,1,1,0),#shape:80*1*1
            nn.ReLU(),
            nn.Conv2d(80,1280,1,1,0),#shape:1280*1*1
            nn.Sigmoid(),
        )
        self.feature_compression=nn.Sequential(#特征压缩
            nn.Conv2d(1280,512,1,1,0),#shape:512*20*20
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(512,256,1,1,0),#shape:256*20*20
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.spatial_enhancement=nn.Sequential(#空间特征增强
            nn.Conv2d(256,256,3,1,1,groups=256),#shape:256*20*20
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,1,1,0),#shape:256*20*20
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        #==========3、全局特征提取==========
        self.global_feature=nn.Sequential(
            nn.AdaptiveAvgPool2d((2*2)),#shape:256*4*4
            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(256*4*4,512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        #==========4、多任务输出头==========
        self.box_head=nn.Sequential(
            nn.Linear(64,4),
            nn.Sigmoid(),
        )
        self.cls_head=nn.Sequential(
            nn.Linear(64,3),
        )
    def forward(self,x):
        features=self.extract(x)
        attention_weights=self.channel_attention(features)
        attended_features=features*attention_weights
        compressed_features=self.feature_compression(attended_features)
        enhanced_features=self.spatial_enhancement(compressed_features)
        global_features=self.global_feature(enhanced_features)
        box=self.box_head(global_features)
        cls=self.cls_head(global_features)
        y=torch.cat((box,cls),dim=1)#torch.cat():延长第dim个维度，dim=1即延长列
        return y
#输出：(x_center,y_center,width,height,class0,class1,class2)

if __name__=='__main__':
    model=MyModel()
    train_dataset = YoloDataset("./datasets/animals_yolo/images/train",
                                "./datasets/animals_yolo/labels/train",
                                img_transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((img_size, img_size)),
                                ]),
                                label_transform=None)
    img,target=train_dataset[0]
    img=img.unsqueeze(0)
    # print(img.shape)
    output=model(img)
    print(output.shape)
    # a=torch.tensor([[1,2],[3,4]])
    # b=torch.tensor([[5,6],[7,8]])
    # print(a)
    # print(b)
    # print(torch.cat((a,b),dim=1))