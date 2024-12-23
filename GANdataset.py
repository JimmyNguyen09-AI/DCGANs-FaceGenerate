from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose,ToTensor,Resize,Normalize,CenterCrop
def data_prepare(dataset_path,image_size = 64):
    transform = Compose([
        Resize(image_size),
        CenterCrop(image_size),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = ImageFolder(root = dataset_path,transform=transform)
    return dataset
if __name__ == '__main__':
    dataset_path = "./dataset"
    dataset = data_prepare(dataset_path)
    dataloader = DataLoader(dataset,batch_size=64,shuffle = True,num_workers=4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()