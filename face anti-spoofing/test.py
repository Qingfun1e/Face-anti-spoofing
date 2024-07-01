from models.swin_unet import TripUNet
import torchvision.transforms as transforms
from PIL import Image
import torch
if __name__ == "__main__":
    model = TripUNet()
    model.load_state_dict(torch.load(r"F:\dlproj\face anti-spoofing\save\best_model_epoch_24.pth"))
    basemodel = model.net
    basemodel = basemodel.to("cuda")
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image = Image.open(r"F:\dlproj\face anti-spoofing\archive\LCC_FASD\LCC_FASD_training\real\AA5742_id154_s0_15.png").convert("RGB")
    image = transform(image)
    C,H,W = image.shape
    image = image.view(1,C,H,W)
    image = image.to("cuda")
    result = basemodel(image)
    print(result[1])