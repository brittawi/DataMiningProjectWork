import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
from unet_model import UNet
from heart_processing import load_nifti, preprocess_image
from torch.utils.data import DataLoader, Dataset

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

        max_size = max(images.shape[2], labels.shape[2])

        if images.shape[2] < max_size:
            images = torch.nn.functional.pad(images, (0, max_size - images.shape[2]))
        if labels.shape[2] < max_size:
            labels = torch.nn.functional.pad(labels, (0, max_size - labels.shape[2]))

        images = F.interpolate(images, size=(32, 32, 32), mode='trilinear', align_corners=False)

        if self.transform:
            images = self.transform(images)
            labels = self.transform(labels)
            image = self.transform(image)
            label = self.transform(label)

        return images, labels
    
def custom_collate_fn(batch):
    """
    Custom collate function that pads all images and labels to the maximum height of the batch.

    This function is used as the collate_fn argument to the DataLoader constructor.
    It takes a list of tuples, where each tuple contains an image and a label.
    It returns a tuple of two tensors, one containing the padded images and the other containing the padded labels.

    Parameters:
        batch (list): A list of tuples, where each tuple contains an image and a label.

    Returns:
        tuple: A tuple of two tensors, one containing the padded images and the other containing the padded labels.
    """
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    max_size = max(image.shape[2] for image in images)
    padded_images = [torch.nn.functional.pad(torch.from_numpy(image), (0, max_size - image.shape[2])) for image in images]
    padded_labels = [torch.nn.functional.pad(torch.from_numpy(label), (0, max_size - label.shape[2])) for label in labels]
    
    return torch.stack(padded_images, 0), torch.stack(padded_labels, 0)

def train_UNet(image_paths, label_paths, num_epochs=25, batch_size=16):
    """
    Train the UNet model on the given dataset.

    Parameters:
        image_paths (list): List of paths to the training images
        label_paths (list): List of paths to the labels of the training images
        num_epochs (int, optional): Number of epochs to train. Defaults to 25.
        batch_size (int, optional): Batch size. Defaults to 16.
    """
    train_dataset = MedicalImageDataset(image_paths, label_paths, transform=preprocess_image)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                images = images.float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
            val_loss /= len(train_loader)
            print(f'Validation Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return model

# test_dataset = MedicalImageDataset(test_image_paths, test_label_paths, transform=preprocess_image)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)