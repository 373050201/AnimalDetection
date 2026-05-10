import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from yolo_dataset import YoloDataset



def calculate_normalization(img_size):#计算数据集的均值与标准差
    train_dataset = YoloDataset("datasets/animals_yolo/images/train",
                                "datasets/animals_yolo/labels/train",
                                img_transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((img_size, img_size)),
                                ]),
                                label_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=8)

    total_pixel = len(train_dataset) * img_size * img_size  # 像素数
    sum_r = sum_g = sum_b = 0.0  # 像素和
    sum_sq_r = sum_sq_g = sum_sq_b = 0.0  # 像素平方之和

    for imgs, _ in train_loader:
        sum_r += torch.sum(imgs[:, 0, :, :])
        sum_g += torch.sum(imgs[:, 1, :, :])
        sum_b += torch.sum(imgs[:, 2, :, :])

        sum_sq_r += torch.sum(imgs[:, 0, :, :] ** 2)
        sum_sq_g += torch.sum(imgs[:, 1, :, :] ** 2)
        sum_sq_b += torch.sum(imgs[:, 2, :, :] ** 2)

    mean_r = sum_r / total_pixel  # 期望公式：EX=像素和/像素数
    mean_g = sum_g / total_pixel
    mean_b = sum_b / total_pixel

    std_r = torch.sqrt(sum_sq_r / total_pixel - mean_r ** 2)  # 标准差公式：sqrt(E(X^2)-(EX)^2)
    std_g = torch.sqrt(sum_sq_g / total_pixel - mean_g ** 2)
    std_b = torch.sqrt(sum_sq_b / total_pixel - mean_b ** 2)

    print(f"三通道均值分别为：[{mean_r.item():.4f},{mean_g.item():.4f},{mean_b.item():.4f}]")
    print(f"三通道标准差分别为：[{std_r.item():.4f},{std_g.item():.4f},{std_b.item():.4f}]")



def calculate_IOU(box1,box2):
    x_c1, y_c1, w_1, h_1 = box1
    x_c2, y_c2, w_2, h_2 = box2

    x1 = x_c1 - 0.5 * w_1
    y1 = y_c1 - 0.5 * h_1
    x2 = x_c1 + 0.5 * w_1
    y2 = y_c1 + 0.5 * h_1
    x3 = x_c2 - 0.5 * w_2
    y3 = y_c2 - 0.5 * h_2
    x4 = x_c2 + 0.5 * w_2
    y4 = y_c2 + 0.5 * h_2

    x_min = max(x1, x3)
    y_min = max(y1, y3)
    x_max = min(x2, x4)
    y_max = min(y2, y4)

    S1 = (x4 - x3) * (y4 - y3)
    S2 = (x2 - x1) * (y2 - y1)
    S = (x_max - x_min) * (y_max - y_min)
    IOU = S / (S1 + S2 - S)
    print(f"IOU为：{IOU:.6f}")



if __name__ == "__main__":
    calculate_normalization(img_size=640)
    calculate_IOU(torch.tensor([0.5,0.5,0.5,0.5]),torch.tensor([0.3,0.3,0.3,0.3]))