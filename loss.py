import torch
from torch import nn



class Loss(nn.Module):
    def __init__(self):
        super(Loss,self).__init__()
        self.location_loss=nn.SmoothL1Loss()#平滑L1损失，用于计算位置损失
        self.class_loss=nn.CrossEntropyLoss()#交叉熵损失，用于计算类别损失，包含了softmax转化为概率的过程
#predict/target：batch_size行7列，列：(x_center,y_center,width,height,class0,class1,class2)
    def forward(self,predict,target):
        predict_location=predict[:,0:4]
        predict_class=predict[:,4:7]
        target_location=target[:,0:4]
        target_class=target[:,4:7]
        lct_loss=self.location_loss(predict_location,target_location)#位置损失
        cls_loss=self.class_loss(predict_class,target_class)#类别损失
        return lct_loss,cls_loss

if __name__ == '__main__':
    predict=torch.tensor([
        [0.4,0.3,0.2,0.1,9.0,-1.4,0.3],
        [0.3,0.2,0.1,0.5,1.0,0.3,4.0]
    ])
    target=torch.tensor([
        [0.3,0.2,0.1,0.2,1.0,0.0,0.0],
        [0.4,0.2,0.1,0.2,0.0,1.0,0.0]
    ])
    loss=Loss()
    print(loss(predict,target))