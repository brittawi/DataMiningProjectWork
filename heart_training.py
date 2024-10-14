import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from unet_model import UNet
from heart_processing import load_nifti, preprocess_image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from torchvision import transforms

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        """
        Parameters:
            image_paths (list): List of paths to the training images
            label_paths (list): List of paths to the labels of the training images
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns the image and label at the given index.

        Parameters:
            idx (int): The index of the sample to return.

        Returns:
            tuple: A tuple containing the image and label as numpy arrays.
        """
        images = load_nifti(self.image_paths[idx])
        labels = load_nifti(self.label_paths[idx])

        images = torch.from_numpy(images).unsqueeze(0).unsqueeze(0)
        labels = torch.from_numpy(labels).unsqueeze(0).unsqueeze(0)

        max_size = max(images.shape[3], labels.shape[3])

        if images.shape[3] < max_size:
            images = torch.nn.functional.pad(images, (0, max_size - images.shape[3]))
        if labels.shape[3] < max_size:
            labels = torch.nn.functional.pad(labels, (0, max_size - labels.shape[3]))

        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        images = normalize(images.squeeze(0))
        images = images.unsqueeze(0)

        return images, labels
    
def custom_collate_fn(batch):
    """
    Custom collate function for loading and padding medical images and labels.

    The function first loads the images and labels from the given batch. It then
    chunks the images and labels into chunks of size `chunk_size`. For each chunk,
    the function determines the maximum size of the images and labels in the
    chunk and then pads the images and labels to this maximum size using the
    `F.pad` function. The padded images and labels are then stacked into a single
    tensor and returned.

    Args:
        batch (list): A list of tuples, where each tuple contains an image and
            label as tensors.

    Returns:
        tuple: A tuple containing the padded images and labels as tensors.
    """
    images, labels = zip(*batch)
    chunk_size = 4
    padded_images = []
    padded_labels = []

    for i in range(0, len(images), chunk_size):
        chunk_images = images[i:i+chunk_size]
        chunk_labels = labels[i:i+chunk_size]

        max_size = max(max(image.shape[2], image.shape[3], image.shape[4]) for image in chunk_images)
        max_label_size = max(max(label.shape[2], label.shape[3], label.shape[4]) for label in chunk_labels)

        padded_chunk_images = []
        padded_chunk_labels = []

        for image in chunk_images:
            padding = (0, 0, 0, 0, 
                       (max_size - image.shape[2]) // 2, (max_size - image.shape[2]) // 2 + (max_size - image.shape[2]) % 2,
                       (max_size - image.shape[3]) // 2, (max_size - image.shape[3]) // 2 + (max_size - image.shape[3]) % 2,
                       (max_size - image.shape[4]) // 2, (max_size - image.shape[4]) // 2 + (max_size - image.shape[4]) % 2)
            padded_image = F.pad(image, padding)
            padded_chunk_images.append(padded_image)

        for label in chunk_labels:
            padding = (0, 0, 0, 0, 
                       (max_label_size - label.shape[2]) // 2, (max_label_size - label.shape[2]) // 2 + (max_label_size - label.shape[2]) % 2,
                       (max_label_size - label.shape[3]) // 2, (max_label_size - label.shape[3]) // 2 + (max_label_size - label.shape[3]) % 2,
                       (max_label_size - label.shape[4]) // 2, (max_label_size - label.shape[4]) // 2 + (max_label_size - label.shape[4]) % 2)
            padded_label = F.pad(label, padding)
            padded_chunk_labels.append(padded_label)

        padded_chunk_images = torch.stack(padded_chunk_images)
        padded_chunk_labels = torch.stack(padded_chunk_labels)

        padded_images.append(padded_chunk_images)
        padded_labels.append(padded_chunk_labels)

    padded_images = torch.cat(padded_images, dim=0)
    padded_labels = torch.cat(padded_labels, dim=0)

    return padded_images, padded_labels

def train_UNet_kfold(image_paths, label_paths, num_epochs=25, batch_size=1, k=5):
    """
    Train the UNet model on the given dataset using 5-fold cross-validation.

    Parameters:
        image_paths (list): List of paths to the training images
        label_paths (list): List of paths to the labels of the training images
        num_epochs (int, optional): Number of epochs to train. Defaults to 25.
        batch_size (int, optional): Batch size. Defaults to 16.
        k (int, optional): Number of folds for cross-validation. Defaults to 5.
    """

    kf = KFold(n_splits=k, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses = []
    val_losses = []

    for fold, (train_index, val_index) in enumerate(kf.split(image_paths)):
        print(f'Fold {fold+1}/{k}')
        train_image_paths = [image_paths[i] for i in train_index]
        val_image_paths = [image_paths[i] for i in val_index]
        train_label_paths = [label_paths[i] for i in train_index]
        val_label_paths = [label_paths[i] for i in val_index]

        train_dataset = MedicalImageDataset(train_image_paths, train_label_paths, transform=preprocess_image)
        val_dataset = MedicalImageDataset(val_image_paths, val_label_paths, transform=preprocess_image)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        model = UNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                images = images.float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                val_loss = 0
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    images = images.float()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
                print(f'Validation Loss: {val_loss:.4f}')

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold{fold+1}.pth')

        train_losses.append(best_loss)

    return train_losses

# test_dataset = MedicalImageDataset(test_image_paths, test_label_paths, transform=preprocess_image)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)