from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WasteDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.df = pd.read_csv(csv_file)
        self.transform = transform

        self.classes = sorted(
            self.df["label"].unique()
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name
            in enumerate(self.classes)
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = (
            PROJECT_ROOT
            / row["filepath"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        label_name = row["label"]

        label = self.class_to_idx[
            label_name
        ]

        if self.transform:
            image = self.transform(image)

        return image, label