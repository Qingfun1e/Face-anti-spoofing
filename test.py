import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F
from models.swin_unet import TripUNet
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def to_tensor(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(np.array(image))
    image = image.view(1, image.shape[0], image.shape[1], image.shape[2])
    return image

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripUNet()
    model.load_state_dict(torch.load("save/1/best_model_epoch_6.pth"))  # 替换为你的模型权重路径
    model = model.net.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_path = r"F:\dlproj\face anti-spoofing\archive\test"
    real_image_path = [os.path.join(test_path + r"\real", item) for item in os.listdir(test_path + r"\real")]
    spoofing_image_path = [os.path.join(test_path + r"\spoofing", item) for item in os.listdir(test_path + r"\spoofing")]
    real_image_label = [1] * len(real_image_path)
    spoofing_image_label = [0] * len(spoofing_image_path)

    print("~~~~~~~~~~~real testing~~~~~~~~~~~~")
    real_test_label = []
    idx = 0
    for image_path in real_image_path:
        image = to_tensor(image_path, transform=transform).to(device)
        regression, classification, _ = model(image)
        reg = regression[0].detach().cpu().permute(1, 2, 0).numpy()
        reg = (reg * 0.5 + 0.5)
        reg = np.clip(reg, 0, 1)
        reg = (reg * 255).astype(np.uint8)
        cv2.imwrite(f'save/real_reg_{idx}.jpg', reg)
        cla = list(F.softmax(classification[0], dim=-1).cpu().detach().numpy())
        label = cla.index(max(cla))
        real_test_label.append(label)
        idx += 1
    real_accuracy = accuracy_score(real_image_label, real_test_label)
    print(real_accuracy)

    print("~~~~~~~~~spoofing testing~~~~~~~~~~")
    spoofing_test_label = []
    idx = 0
    for image_path in spoofing_image_path:
        image = to_tensor(image_path, transform=transform).to(device)
        regression, classification, _ = model(image)
        reg = regression[0].detach().cpu().permute(1, 2, 0).numpy()
        reg = (reg * 0.5 + 0.5)
        reg = np.clip(reg, 0, 1)
        reg = (reg * 255).astype(np.uint8)
        cv2.imwrite(f'save/spoofing_reg_{idx}.jpg', reg)
        cla = list(F.softmax(classification[0], dim=-1).cpu().detach().numpy())
        label = cla.index(max(cla))
        spoofing_test_label.append(label)
        idx += 1
    spoofing_accuracy = accuracy_score(spoofing_image_label, spoofing_test_label)
    print(spoofing_accuracy)

    # 计算混淆矩阵
    y_true = real_image_label + spoofing_image_label
    y_pred = real_test_label + spoofing_test_label
    cm = confusion_matrix(y_true, y_pred)

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # 计算评价指标
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{cm}')
