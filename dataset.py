from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import matplotlib.pyplot as plt

class WeaponsDataset(Dataset):

    def __init__(self):
        self.samples = []
        img_folder_path = "./data/Dark_Souls_1_Weapons"
        image_paths = [join(img_folder_path, f) for f in listdir(img_folder_path)]
        for path in image_paths:
            img = Image.open(path)
            image_tensor = pil_to_tensor(img)
            self.samples.append(image_tensor)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    dataset = WeaponsDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    examples = iter(dataloader)
    sample = next(examples).view(4, 90, 80)
    image = np.transpose(sample.detach(), (1, 2, 0))
    plt.imshow(image)
    plt.show()