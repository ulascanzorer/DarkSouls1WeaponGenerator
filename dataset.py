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

    def __init__(self):
        self.samples = []
        self.transform = transforms.Compose(
        [transforms.Resize((64, 64)),
        transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
        )

        # Get Dark Souls 1 weapons
        ds1_img_folder_path = "./data/Dark_Souls_1_Weapons"
        ds1_image_paths = [join(ds1_img_folder_path, f) for f in listdir(ds1_img_folder_path)]
        for path in ds1_image_paths:
            img = Image.open(path)
            image_tensor = pil_to_tensor(img)
            image_tensor = image_tensor / 255

           
            image_tensor = self.transform(image_tensor)

            # image_tensor = F.pad(input=image_tensor, pad=(0, 10, 0, 0, 0, 0), mode="constant", value=-1)

            self.samples.append(image_tensor)

        # Get Dark Souls 2 weapons
        ds2_img_folder_path = "./data/Dark_Souls_2_Weapons"
        ds2_image_paths = [join(ds2_img_folder_path, f) for f in listdir(ds2_img_folder_path)]
        for path in ds2_image_paths:
            img = Image.open(path)
            image_tensor = pil_to_tensor(img)
            # print(path)
            # print(image_tensor.shape)
            if image_tensor.shape[0] != 4:
                continue
            image_tensor = image_tensor / 255

            image_tensor = self.transform(image_tensor)

            self.samples.append(image_tensor)   


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    dataset = WeaponsDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    examples = iter(dataloader)
    samples = next(examples)
    sample = samples[0]
    print(len(dataloader))
    # print(sample.shape, sample.dtype)
    # sample = sample / 255
    
    # sample = transform(sample)
    # print(sample.dtype)
    print(sample.shape)

    image = np.transpose(sample.detach(), (1, 2, 0))
    image = image / 2 + 0.5
    plt.imshow(image)
    plt.show()