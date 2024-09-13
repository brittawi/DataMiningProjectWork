import nibabel as nib
import os
import matplotlib.pyplot as plt

def load_and_plot(data_type, data_list, base_path):
    num_samples = min(5, len(data_list))
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, num_samples * 6))
    
    for i in range(num_samples):
        item = data_list[i]
        image_path = os.path.join(base_path, item['image'])
        label_path = os.path.join(base_path, item['label']) if 'label' in item else None
        
        image = nib.load(image_path)
        image_data = image.get_fdata()
        
        if label_path:
            label = nib.load(label_path)
            label_data = label.get_fdata()
        
        if image_data.ndim == 4:
            image_data = image_data[:, :, :, 0]
        if label_path and label_data.ndim == 4:
            label_data = label_data[:, :, :, 0]

        middle_slice_img = image_data[:, :, image_data.shape[2] // 2]
        middle_slice_label = label_data[:, :, label_data.shape[2] // 2] if label_path else None

        axs[i, 0].imshow(middle_slice_img.T, cmap='gray', origin='lower')
        axs[i, 0].set_title(f'{data_type} Image {i+1}')
        axs[i, 0].axis('off')

        if middle_slice_label is not None:
            axs[i, 1].imshow(middle_slice_img.T, cmap='gray', origin='lower')
            axs[i, 1].imshow(middle_slice_label.T, cmap='jet', alpha=0.5, origin='lower')
            axs[i, 1].set_title(f'{data_type} Image & Label {i+1}')
        else:
            axs[i, 1].imshow(middle_slice_img.T, cmap='gray', origin='lower')
            axs[i, 1].set_title(f'{data_type} Image {i+1} (No Label)')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()