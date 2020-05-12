import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class YelpDataset(Dataset):
    """Yelp Classification dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # return 5000
        return len(self.img_df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.img_df.iloc[idx, 2])
        image = Image.open(img_name)
        # temp = io.imread(img_name)
        # labels = self.img_df.iloc[idx, 1][2:-2].split(' ')
        labels = self.img_df.iloc[idx, 1][1:-1].split(',')
        labels = [l.strip()[1] for l in labels]

        labels = list(map(int, labels))
        # labels = np.array(labels).astype(np.int)
        target = np.zeros((9,))
        target[labels] = 1
        # sample = {'image': image, 'labels': target}
        # plt.imshow(image)
        # plt.show()

        if self.transform:
            image = self.transform(image)
        # print(image.shape)
        # return index image and target
        # return the type which pytorch can use to train the model;
        return (image, target)


if __name__ == '__main__':
    train_transform = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])
    dataset = YelpDataset("../DATA.csv", "../../training_gallary", train_transform)
    img, target = dataset[0]
    print(img.shape)
    print(len(dataset))
