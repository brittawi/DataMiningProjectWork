import json
import nibabel as nib
import numpy as np
from skimage.transform import resize
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import torch

def load_nifti(file_path):
    """
    Load a NIfTI file and return the image as a numpy array.

    Parameters:
        file_path (str): Path to the NIfTI file.

    Returns:
        img_data (numpy.ndarray): The image as a numpy array.
    """
    nifti_img = nib.load(file_path)

    img_data = nifti_img.get_fdata()

    return img_data

def preprocess_image(image, resize=None, crop=None, augment=False):
    """
    Preprocess a NIfTI image.

    Parameters:
        image (numpy.ndarray): The image to preprocess.
        resize (tuple, optional): The size to resize the image to.
        crop (tuple, optional): The size to crop the image to.
        augment (bool, optional): If true, augment the image with random rotation, flipping and cropping.

    Returns:
        image (numpy.ndarray): The preprocessed image.
    """
    print("Starting preprocessing with image shape:", image.shape)
    image = (image - np.mean(image)) / np.std(image)

    if resize is not None:
        image = resize_image(image, resize)
        print("Image shape after resizing:", image.shape)

    if crop is not None:
        print("Image shape before cropping:", image.shape)
        image = crop_image(image, crop)
        print("Image shape after cropping:", image.shape)

    if augment:
        image = augment_image(image)
        print("Image shape after augmentation:", image.shape)

    print("Finished preprocessing, returning image with shape:", image.shape)
    return image

def resize_image(image, resize_size):
    """Resize the image to the specified size."""
    return resize(image, resize_size, anti_aliasing=True)

def crop_image(image, crop):
    """Crop the image to the specified size."""
    if len(image.shape) == 3:
        z, y, x = image.shape
        z_start = (z - crop[0]) // 2
        y_start = (y - crop[1]) // 2
        x_start = (x - crop[2]) // 2
        return image[z_start:z_start + crop[0],
                     y_start:y_start + crop[1],
                     x_start:x_start + crop[2]]
    elif len(image.shape) == 4:
        c, z, y, x = image.shape
        z_start = (z - crop[0]) // 2
        y_start = (y - crop[1]) // 2
        x_start = (x - crop[2]) // 2
        return image[:, z_start:z_start + crop[0],
                     y_start:y_start + crop[1],
                     x_start:x_start + crop[2]]
    else:
        raise ValueError("Unsupported image shape: {}".format(image.shape))

def augment_image(image):
    """Augment the image with random rotation, flipping, and cropping."""
    image = random_rotation(image, rg=30)
    image = random_flip(image)
    image = random_crop(image, crop_shape=(96, 96, 96))
    return image

def random_rotation(image, rg):
    """Rotate the image by a random angle between -rg and rg."""
    rot_angle = np.random.uniform(-rg, rg)
    rotated_image = rotate(image, rot_angle, axes=(0,1), reshape=False)
    print("Image shape after rotation:", rotated_image.shape)
    return rotated_image

def random_flip(image):
    """Flip the image along the x and y axes."""
    flipped_image = image[:, ::-1, ::-1]
    print("Image shape after flipping:", flipped_image.shape)
    return flipped_image

def random_crop(image, crop_shape):
    """Crop the image to the specified shape."""
    z, y, x = image.shape
    if (z < crop_shape[0]) or (y < crop_shape[1]) or (x < crop_shape[2]):
        raise ValueError("Crop shape {} is larger than image shape {}".format(crop_shape, image.shape))

    z_start = np.random.randint(0, z - crop_shape[0] + 1)
    y_start = np.random.randint(0, y - crop_shape[1] + 1)
    x_start = np.random.randint(0, x - crop_shape[2] + 1)

    return image[z_start:z_start + crop_shape[0],
                 y_start:y_start + crop_shape[1],
                 x_start:x_start + crop_shape[2]]

def load_and_preprocess(image_path, label_path):
    """
    Load and preprocess both image and label.

    Parameters:
        image_path (str): Path to the image file.
        label_path (str): Path to the label file.

    Returns:
        image_tensor (torch.Tensor): The preprocessed image as a tensor.
        label_tensor (torch.Tensor): The preprocessed label as a tensor.
    """
    image = load_nifti(image_path)
    label = load_nifti(label_path)
    
    print("Image shape before cropping:", image.shape)
    print("Label shape before cropping:", label.shape)

    preprocessed_image = preprocess_image(image, resize=(96, 96, 96), augment=True)

    image_tensor = torch.tensor(preprocessed_image.copy(), dtype=torch.float32)
    label_tensor = torch.tensor(label.copy(), dtype=torch.long)

    return image_tensor, label_tensor

def plot_overlay_preprocessed_image_and_label(json_path):
    """Overlay the preprocessed label on the preprocessed MRI image.

    This function loads the dataset JSON file, takes 5 random samples from it, loads the
    images and labels, preprocesses them, and then overlays the label on the image.

    Parameters:
        json_path (str): The path to the dataset JSON file.

    Returns:
        None
    """
    
    with open(json_path) as f:
        data = json.load(f)
    
    sample_list = np.random.choice(data['training'], 5, replace=False)
    
    fig, axes = plt.subplots(len(sample_list), 2, figsize=(10, len(sample_list) * 5))
    
    for i, sample in enumerate(sample_list):
        image_path = sample['image']
        label_path = sample['label']
        
        image_tensor, label_tensor = load_and_preprocess(image_path, label_path)
        
        slice_index = np.random.choice(image_tensor.shape[2])
        
        image = image_tensor.numpy()
        label = label_tensor.numpy()
        
        image_slice = image[:, :, slice_index]
        label_slice = label[:, :, slice_index]
        
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        
        unique_labels = np.unique(label_slice)
        print(f'Slice {slice_index} - Unique label values: {unique_labels}')
        
        if len(unique_labels) == 1 and unique_labels[0] == 0:
            print(f'Slice {slice_index} contains no segmentation. Skipping overlay.')
        
        axes[i, 0].imshow(image_slice, cmap='gray')
        axes[i, 0].set_title(f'Original MRI Slice {slice_index}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(image_slice, cmap='gray')
        
        if len(unique_labels) > 1 or unique_labels[0] != 0:
            axes[i, 1].imshow(label_slice, cmap='jet', alpha=0.4)
            axes[i, 1].set_title(f'Overlayed Label Slice {slice_index}')
        else:
            axes[i, 1].set_title(f'No Segmentation in Slice {slice_index}')
        
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def load_dataset(json_path):
    """Load all images and labels from the dataset JSON file.

    This function reads the dataset JSON file and loads all the images and labels
    into memory.

    Parameters:
        json_path (str): The path to the dataset JSON file.

    Returns:
        tuple: A tuple containing two lists. The first list contains the image
            tensors and the second list contains the label tensors.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    labels = []

    for sample in data['training']:
        image_path = sample['image']
        label_path = sample['label']

        image_tensor, label_tensor = load_and_preprocess(image_path, label_path)
        
        images.append(image_tensor)
        labels.append(label_tensor)
    
    return images, labels