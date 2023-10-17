from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F


class WeaponsDataset(Dataset):

    def __init__(self, transform=None):
        self.samples = []

        self.transform = transform
        img_folder_path = "./data/Dark_Souls_1_Weapons"
        image_paths = [join(img_folder_path, f) for f in listdir(img_folder_path)]
        for path in image_paths:
            img = Image.open(path)
            image_tensor = pil_to_tensor(img)
            image_tensor = image_tensor / 255

            if self.transform is not None:
                image_tensor = self.transform(image_tensor)

            # image_tensor = F.pad(input=image_tensor, pad=(0, 10, 0, 0, 0, 0), mode="constant", value=-1)

            self.samples.append(image_tensor)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
        )
    dataset = WeaponsDataset(transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    examples = iter(dataloader)
    samples = next(examples)
    sample = samples[0]

    # print(sample.shape, sample.dtype)
    # sample = sample / 255
    
    # sample = transform(sample)
    # print(sample.dtype)
    print(sample.shape)

    image = np.transpose(sample.detach(), (1, 2, 0))
    image = image / 2 + 0.5
    plt.imshow(image)
    plt.show()