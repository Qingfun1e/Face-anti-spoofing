import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from models.swin_unet import TripUNet
import torch.backends.cudnn as cudnn
from utils.loss_fn import TotalLoss
from utils.data_handler import TripletDataset
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

cudnn.benchmark = True


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model...')
        torch.save(model.state_dict(), f'save/best_model_epoch_{epoch}.pth')
        self.val_loss_min = val_loss


def train_tl(model, criterion, optimizer, dataloaders, num_epochs, scheduler=None, early_stopping=None):
    step = 0
    total_loss = []
    max_grad_norm = 1.0  # 设置梯度剪枝的阈值

    scaler = GradScaler()

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                running_loss = 0.0
                model.train()
                progress_bar = tqdm(dataloaders[phase], desc=f"Epoch {epoch + 1}/{num_epochs} (Training)")

                for anchors, positives, negatives, labels in progress_bar:
                    anchors = anchors.cuda()
                    positives = positives.cuda()
                    negatives = negatives.cuda()
                    labels = labels.cuda()

                    optimizer.zero_grad()

                    with autocast():
                        regression, classification, feat = model(anchors, positives, negatives)
                        loss = criterion(regression, classification, feat, labels)

                    scaler.scale(loss).backward()

                    # 梯度剪枝
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    total_loss.append(loss.item())
                    step += 1

                    progress_bar.set_postfix(loss=loss.item())
                    progress_bar.update(1)

                if scheduler is not None:
                    scheduler.step(running_loss)

            else:  # validation phase
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for anchors, positives, negatives, labels in dataloaders[phase]:
                        anchors = anchors.cuda()
                        positives = positives.cuda()
                        negatives = negatives.cuda()
                        labels = labels.cuda()

                        with autocast():
                            regression, classification, feat = model(anchors, positives, negatives)
                            loss = criterion(regression, classification, feat, labels)

                        val_loss += loss.item()

                val_loss /= len(dataloaders[phase])

                if early_stopping is not None:
                    early_stopping(val_loss, model, epoch)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        return

                torch.save(model.state_dict(), "save/{}.pth".format(epoch))

        # 每个epoch结束后绘制损失曲线
        plt.plot(total_loss, label='Training Loss')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.savefig('loss_curve_epoch.png')
        plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripUNet()
    criterion = TotalLoss().to(device)  # Move criterion to the device
    dataset = TripletDataset()

    # 按照9:1分割数据集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2),
        'val': DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    }

    optimizer = optim.AdamW(params=model.parameters(), lr=1e-4)

    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 早停法
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_tl(model, criterion, optimizer, dataloaders, num_epochs=50, scheduler=scheduler,
             early_stopping=early_stopping)
