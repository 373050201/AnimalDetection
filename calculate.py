import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from yolo_dataset import YoloDataset



img_size=640
train_dataset=YoloDataset("datasets/animals_yolo/images/train",
                            "datasets/animals_yolo/labels/train",
                            img_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((img_size, img_size)),
                            ]),
                            label_transform=None)
train_loader=DataLoader(train_dataset,batch_size=8)



total_pixel=len(train_dataset)*img_size*img_size#像素数
sum_r=sum_g=sum_b=0.0#像素和
sum_sq_r=sum_sq_g=sum_sq_b=0.0#像素平方之和

for imgs,_ in train_loader:
    sum_r+=torch.sum(imgs[:,0,:,:])
    sum_g+=torch.sum(imgs[:,1,:,:])
    sum_b+=torch.sum(imgs[:,2,:,:])

    sum_sq_r+=torch.sum(imgs[:,0,:,:]**2)
    sum_sq_g+=torch.sum(imgs[:,1,:,:]**2)
    sum_sq_b+=torch.sum(imgs[:,2,:,:]**2)

mean_r=sum_r/total_pixel#期望公式：EX=像素和/像素数
mean_g=sum_g/total_pixel
mean_b=sum_b/total_pixel

std_r=torch.sqrt(sum_sq_r/total_pixel-mean_r**2)#标准差公式：sqrt(E(X^2)-(EX)^2)
std_g=torch.sqrt(sum_sq_g/total_pixel-mean_g**2)
std_b=torch.sqrt(sum_sq_b/total_pixel-mean_b**2)

print("三通道均值分别为：[{:.4f},{:.4f},{:.4f}]".format(mean_r.item(),mean_g.item(),mean_b.item()))
print("三通道标准差分别为：[{:.4f},{:.4f},{:.4f}]".format(std_r.item(),std_g.item(),std_b.item()))