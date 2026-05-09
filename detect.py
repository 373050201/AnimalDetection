import torch
from PIL import Image
from model import MyModel
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt



def load_image(path):
    img=Image.open(path).convert("RGB")
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    return transform(img)



print("正在加载模型...")
myModel=MyModel()
myModel=myModel.cuda()
state_dict = torch.load("./model_best.pth")
myModel.load_state_dict(state_dict)
img_path= "detect/detect.png"
label_path= "detect/detect.txt"

print("正在预测...")
img=load_image(img_path)
img=img.unsqueeze(0)
img=img.cuda()
output=myModel(img)

pred_cls=torch.max(output[0][4:7],dim=0).indices
classes=["cat","dog","bird"]
print(f"预测类别：{pred_cls}——{classes[pred_cls]}")

x_c=output[0][0].item()
y_c=output[0][1].item()
w=output[0][2].item()
h=output[0][3].item()
print(f"预测位置：{x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}")
with open(label_path, 'w') as f:
    f.write(f"{pred_cls} {x_c:.4f} {y_c:.4f} {w:.4f} {h:.4f}")

# 绘制矩形框
with open(label_path, 'r') as f:
    lines = f.readlines()
image = cv2.imread(img_path)
h, w = image.shape[:2]
for line in lines:
    cls, cx, cy, bw, bh = map(float, line.strip().split())
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.axis('off')
plt.show()