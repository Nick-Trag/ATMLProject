import os

from PIL import Image
from torch.utils.data import Dataset

label_map = {
    "NORMAL": 0,
    "CNV": 1,
    "DME": 2,
    "DRUSEN": 3
}


class OCTDataset(Dataset):
    def __init__(self, root_directory="../data/OCT2017", transform=None, mode='train'):
        self.root_directory = root_directory
        self.transform = transform
        # Count the number of files in order to reserve the memory for the array
        num_files = sum([len(files) for __, __, files in os.walk(os.path.join(root_directory, mode))])
        self.files = ["" for __ in range(num_files)]
        self.labels = [4 for __ in range(num_files)]
        counter = 0
        # Save all the filenames in an array and all the labels in another parallel array
        for directory in os.listdir(os.path.join(root_directory, mode)):
            # Get full path
            directory_full = os.path.join(root_directory, mode, directory)
            for file in os.listdir(directory_full):
                file = os.path.join(directory_full, file)
                self.files[counter] = file
                self.labels[counter] = label_map[directory]
                counter += 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(Image.open(self.files[item])), self.labels[item]
        else:
            return Image.open(self.files[item]), self.labels[item]
