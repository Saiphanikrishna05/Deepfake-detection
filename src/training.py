import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from models.visual_model import DeepfakeDetector
from src.preprocessing import DeepfakeDataset

def create_train_loader(root='data/train', batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    real = DeepfakeDataset(f'{root}/real', label=0, transform=transform)
    fake = DeepfakeDataset(f'{root}/fake', label=1, transform=transform)
    ds = ConcatDataset([real, fake])
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

def train(epochs=1, lr=1e-4, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector().to(device)
    loader = create_train_loader()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.float().unsqueeze(1).to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0:
                print(f'Epoch {epoch+1}, batch {i}, loss {loss.item():.4f}')
    torch.save(model.state_dict(), 'outputs/models/visual_baseline.pth')
    return model

if __name__ == '__main__':
    train()
