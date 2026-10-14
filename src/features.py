"""DINOv2 feature extraction (PyTorch) used by train_features.py and the demo app."""
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms

BACKBONES = {
    "dinov2_base": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_large": "vit_large_patch14_dinov2.lvd142m",
}
IMAGE_SIZE = 224

TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_backbone(name, device):
    model = timm.create_model(BACKBONES[name], pretrained=True, num_classes=0, img_size=IMAGE_SIZE)
    return model.eval().to(device)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return TRANSFORM(Image.open(self.paths[index]).convert("RGB"))

@torch.no_grad()
def embed(model, batch, device):
    batch = batch.to(device)
    # Average the embeddings of each image and its mirror image.
    features = (model(batch) + model(torch.flip(batch, dims=[3]))) / 2
    return features.cpu().numpy()

def extract_features(model, paths, device, batch_size=32, num_workers=0):
    # Worker processes are slower here on macOS (each one re-imports TensorFlow and PyTorch).
    loader = torch.utils.data.DataLoader(
        ImageDataset(paths), batch_size=batch_size, num_workers=num_workers
    )
    chunks = []
    for i, batch in enumerate(loader, start=1):
        chunks.append(embed(model, batch, device))
        if i % 20 == 0 or i == len(loader):
            print(f"  {min(i * batch_size, len(loader.dataset))}/{len(loader.dataset)} images", flush=True)
    return np.concatenate(chunks)

def embed_image(model, image, device):
    return embed(model, TRANSFORM(image.convert("RGB"))[None], device)
