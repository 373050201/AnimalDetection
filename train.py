import torch
from torch.utils.data import DataLoader
from loss import Loss
from yolo_dataset import YoloDataset,format_convert
import torchvision.transforms as transforms
from model import MyModel
from torchvision.ops import box_iou



img_size=640
print("正在加载数据集...")
mean=[0.5861,0.5767,0.4773]
std=[0.2547,0.2416,0.2828]
train_dataset=YoloDataset("datasets/animals_yolo/images/train",
                            "datasets/animals_yolo/labels/train",
                            img_transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Resize((img_size, img_size)),
                                transforms.Normalize(mean=mean, std=std)
                            ]),
                            label_transform=format_convert)
train_loader=DataLoader(train_dataset,batch_size=8)
val_dataset=YoloDataset("datasets/animals_yolo/images/val",
                        "datasets/animals_yolo/labels/val",
                        img_transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize((img_size, img_size)),
                            transforms.Normalize(mean=mean, std=std)
                        ]),
                        label_transform=format_convert)
val_loader=DataLoader(val_dataset,batch_size=2)

print("正在初始化模型...")
myModel=MyModel()
myModel=myModel.cuda()
loss=Loss()
loss=loss.cuda()
learning_rate=0.0001
optimizer=torch.optim.Adam(myModel.parameters(),lr=learning_rate)

total_train=0
epoch=100
for i in range(epoch):
    print(f"第{i+1}次训练开始")
    train_lct_loss_sum=0
    train_cls_loss_sum=0
    for imgs,targets in train_loader:
        imgs=imgs.cuda()
        targets=targets.cuda()

        outputs=myModel(imgs)
        train_lct_loss,train_cls_loss=loss(outputs,targets)
        train_lct_loss_sum+=train_lct_loss
        train_cls_loss_sum+=train_cls_loss
        train_sum_loss=train_lct_loss+train_cls_loss
        optimizer.zero_grad()
        train_sum_loss.backward()
        optimizer.step()

        total_train+=1
    print(f"此轮已训练了{total_train}次")
    print(f"训练集lct损失：{train_lct_loss_sum:.6f},训练集cls损失：{train_cls_loss_sum:.6f}")
    print(f"训练集总损失：{train_lct_loss_sum+train_cls_loss_sum:.6f}")

    val_epoch=int(epoch/epoch)#每val_epoch轮评估一次
    if (i+1)%val_epoch==0:
        val_lct_loss_sum=0
        val_cls_loss_sum=0
        total_correct=0#预测正确总数，要求类别预测准确且预测框IOU>0.5
        with torch.no_grad():
            myModel.eval()
            for imgs,targets in val_loader:
                imgs=imgs.cuda()
                targets=targets.cuda()
                outputs=myModel(imgs)#shape:[2,7]
                #计算验证集损失
                val_lct_loss,val_cls_loss=loss(outputs,targets)
                val_lct_loss_sum+=val_lct_loss
                val_cls_loss_sum+=val_cls_loss

                for idx in range(len(outputs)):
                    cls_pred = outputs[idx][4:7]#提取类别，若类别相等则判定预测准确
                    cls_true = targets[idx][4:7]
                    max_idx_pred=torch.max(cls_pred,dim=0).indices.item()#torch.max():按dim方向比较，二维时dim=1即比较每一列
                    max_idx_true=torch.max(cls_true,dim=0).indices.item()
                    box_pred = outputs[idx:idx+1,0:4]#提取预测框，用IOU（交并比）评估，若>0.5则判定预测准确
                    box_true = targets[idx:idx+1,0:4]#shape:1*4
                    iou=box_iou(box_pred,box_true,fmt="cxcywh").item()
                    if max_idx_pred==max_idx_true and iou>0.5:
                        total_correct+=1
        print(f"验证集lct损失：{val_lct_loss_sum:.6f},验证集cls损失：{val_cls_loss_sum:.6f}")
        print(f"验证集总损失：{val_lct_loss_sum+val_cls_loss_sum:.6f}")
        print(f"验证集预测准确率：{total_correct/len(val_dataset):.6f}")
print("训练结束")
torch.save(myModel.state_dict(),f"model_{epoch}.pth")