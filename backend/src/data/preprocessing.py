import os
import cv2
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


class GTSRBDataset(Dataset):
    """
    PyTorch Dataset for GTSRB.

    Expected dataframe columns:
    - FullPath : absolute path to image
    - ClassId  : integer class label
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "FullPath"]
        label = int(self.dataframe.loc[idx, "ClassId"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def load_metadata(base_path: str):
    """
    Load Train.csv and Test.csv from dataset root.
    """
    train_csv_path = os.path.join(base_path, "Train.csv")
    test_csv_path = os.path.join(base_path, "Test.csv")

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    return train_df, test_df


def add_full_paths(train_df: pd.DataFrame, test_df: pd.DataFrame, base_path: str):
    """
    Add absolute FullPath column to train and test dataframes.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["FullPath"] = train_df["Path"].apply(lambda x: os.path.join(base_path, x))
    test_df["FullPath"] = test_df["Path"].apply(lambda x: os.path.join(base_path, x))

    return train_df, test_df


def verify_paths(dataframe: pd.DataFrame, path_column: str = "FullPath") -> bool:
    """
    Check whether all file paths exist.
    """
    return dataframe[path_column].apply(os.path.exists).all()


def get_transforms(img_size: int = 32, augment: bool = False):
    """
    Build image transforms.

    For train:
        augment=True
    For val/test:
        augment=False
    """
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])


def split_train_validation(
    train_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42
):
    """
    Stratified train-validation split using ClassId.
    """
    train_split_df, val_split_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df["ClassId"],
        random_state=random_state
    )

    train_split_df = train_split_df.reset_index(drop=True)
    val_split_df = val_split_df.reset_index(drop=True)

    return train_split_df, val_split_df


def create_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_size: int = 32,
    use_augmentation: bool = True
):
    """
    Create train, validation, and test dataset objects.
    """
    train_transform = get_transforms(img_size=img_size, augment=use_augmentation)
    eval_transform = get_transforms(img_size=img_size, augment=False)

    train_dataset = GTSRBDataset(train_df, transform=train_transform)
    val_dataset = GTSRBDataset(val_df, transform=eval_transform)
    test_dataset = GTSRBDataset(test_df, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


def compute_class_weights(train_df: pd.DataFrame):
    """
    Compute inverse-frequency class weights for imbalanced classification.

    Returns:
        class_weights: torch.FloatTensor
        class_counts : pd.Series
    """
    class_counts = train_df["ClassId"].value_counts().sort_index()
    num_classes = len(class_counts)

    class_weights = 1.0 / class_counts.values
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return class_weights, class_counts


def create_weighted_sampler(train_df: pd.DataFrame, class_counts: pd.Series):
    """
    Create a WeightedRandomSampler for balanced training batches.
    """
    sample_weights = train_df["ClassId"].map(lambda x: 1.0 / class_counts[x]).values
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    train_sampler=None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """
    Create dataloaders for train, validation, and test.

    If train_sampler is provided, train_loader will not shuffle.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def build_data_pipeline(
    base_path: str,
    img_size: int = 32,
    batch_size: int = 32,
    val_size: float = 0.2,
    random_state: int = 42,
    use_augmentation: bool = True,
    use_weighted_sampler: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True
):
    """
    Full end-to-end preprocessing pipeline.

    Returns a dictionary containing:
    - train_df, val_df, test_df
    - train_dataset, val_dataset, test_dataset
    - train_loader, val_loader, test_loader
    - class_weights, class_counts
    - train_sampler
    """
    train_df, test_df = load_metadata(base_path)
    train_df, test_df = add_full_paths(train_df, test_df, base_path)

    if not verify_paths(train_df):
        raise FileNotFoundError("Some training image paths are invalid.")
    if not verify_paths(test_df):
        raise FileNotFoundError("Some test image paths are invalid.")

    train_split_df, val_split_df = split_train_validation(
        train_df=train_df,
        val_size=val_size,
        random_state=random_state
    )

    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df=train_split_df,
        val_df=val_split_df,
        test_df=test_df,
        img_size=img_size,
        use_augmentation=use_augmentation
    )

    class_weights, class_counts = compute_class_weights(train_split_df)

    train_sampler = None
    if use_weighted_sampler:
        train_sampler = create_weighted_sampler(train_split_df, class_counts)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        "train_df": train_split_df,
        "val_df": val_split_df,
        "test_df": test_df,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": class_weights,
        "class_counts": class_counts,
        "train_sampler": train_sampler,
    }


def preprocess(image, img_size=32):
    """
    Preprocess a single image for inference.

    Accepts PIL images, NumPy arrays, or tensors. The evaluation transform
    used elsewhere in the pipeline starts with `ToPILImage()`, so we normalize
    PIL input back to an array here to keep one inference path for the API.
    """
    transform = get_transforms(img_size=img_size, augment=False)
    if hasattr(image, "mode") and hasattr(image, "size"):
        image = np.array(image)
    return transform(image).unsqueeze(0)
