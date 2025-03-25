import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from models.swin_unet import TripUNet
import torch.backends.cudnn as cudnn
from utils.loss_fn import TotalLoss
from utils.data_handler import TripletDataset
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

cudnn.benchmark = True

def train_tl(model, criterion, optimizer, dataloader, num_epochs, scheduler=None):
    step = 0
    loss_list = []
    max_grad_norm = 1.0  # 设置梯度剪枝的阈值

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Training)")

        for anchors, positives, negatives, labels in progress_bar:
            anchors, positives, negatives, labels = anchors.cuda(), positives.cuda(), negatives.cuda(), labels.cuda()

            optimizer.zero_grad()

            with autocast():
                regression, classification, feat = model(anchors, positives, negatives)
                loss = criterion(regression, classification, feat, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print("Loss is NaN or Inf, skipping this batch")
                    torch.cuda.empty_cache()
                    continue

            scaler.scale(loss).backward()

            # 梯度剪枝
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            loss_list.append(loss.item())
            step += 1

            progress_bar.set_postfix(loss=loss.item())
            progress_bar.update(1)

            if scheduler is not None:
                scheduler.step(loss.item())

            # 每次迭代步结束后绘制损失曲线
            plt.plot(loss_list, label='Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend()
            plt.savefig('loss_curve_step.png')
            plt.close()

        # 保存模型
        torch.save(model.state_dict(), f"save/epoch_{epoch}.pth")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripUNet()
    model = model.to(device)
    criterion = TotalLoss().to(device)  # Move criterion to the device
    dataset = TripletDataset()

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-4)  # 降低学习率

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    train_tl(model, criterion, optimizer, dataloader, num_epochs=50, scheduler=scheduler)
