import SimpleITK as sitk
import numpy as np
import scipy.ndimage as ndi
import os
import time
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

mri_files = ['imagesTr/BRATS_%03d.nii.gz' % i for i in range(1, 7)]
seg_files = ['labelsTr/BRATS_%03d.nii.gz' % i for i in range(1, 7)]

for f in mri_files + seg_files:
    if not os.path.exists(f):
        print(f"File not found: {f}")

start_time = time.time()

mr_images = [sitk.ReadImage(f) for f in mri_files]
seg_images = [sitk.ReadImage(f) for f in seg_files]

mri_array = np.array([sitk.GetArrayFromImage(mr) for mr in mr_images])
seg_array = np.array([sitk.GetArrayFromImage(s) for s in seg_images])

print("Time taken to read and convert images:", time.time() - start_time)
print("Initial MRI shape:", mri_array.shape)
print("Initial Segmentation shape:", seg_array.shape)

mri_array = (mri_array - np.min(mri_array)) / (np.max(mri_array) - np.min(mri_array))

def resize_image(image, new_shape, order=1):
    zoom_factors = [new_dim / old_dim for old_dim, new_dim in zip(image.shape, new_shape)]
    return ndi.zoom(image, zoom_factors, order=order, mode='nearest')

mri_resized = np.array([resize_image(mr, (4, 256, 256, 256)) for mr in mri_array])
seg_resized = np.array([resize_image(seg, (256, 256, 256)) for seg in seg_array])

train_indices, val_indices = train_test_split(np.arange(mri_resized.shape[0]), test_size=0.2, random_state=42)

if len(train_indices) == 0:
    print("Not enough samples for training. Adjust test_size.")
else:
    train_mri = mri_resized[train_indices]
    val_mri = mri_resized[val_indices]

    train_seg = seg_resized[train_indices]
    val_seg = seg_resized[val_indices]

    print("Train MRI shape:", train_mri.shape)  # Expected: (num_train_samples, 4, 256, 256, 256)
    print("Validation MRI shape:", val_mri.shape)  # Expected: (num_val_samples, 4, 256, 256, 256)
    print("Train Segmentation shape:", train_seg.shape)  # Expected: (num_train_samples, 256, 256, 256)
    print("Validation Segmentation shape:", val_seg.shape)  # Expected: (num_val_samples, 256, 256, 256)

# Encoder
inputs = keras.layers.Input(shape=(4, 256, 256, 256))
conv1 = keras.layers.Conv3D(8, 3, activation='relu', padding='same')(inputs)
conv1 = keras.layers.Conv3D(8, 3, activation='relu', padding='same')(conv1)
pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
conv2 = keras.layers.Conv3D(16, 3, activation='relu', padding='same')(pool1)
conv2 = keras.layers.Conv3D(16, 3, activation='relu', padding='same')(conv2)
pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
conv3 = keras.layers.Conv3D(32, 3, activation='relu', padding='same')(pool2)
conv3 = keras.layers.Conv3D(32, 3, activation='relu', padding='same')(conv3)
pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
conv4 = keras.layers.Conv3D(64, 3, activation='relu', padding='same')(pool3)
conv4 = keras.layers.Conv3D(64, 3, activation='relu', padding='same')(conv4)
pool4 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)
conv5 = keras.layers.Conv3D(128, 3, activation='relu', padding='same')(pool4)
conv5 = keras.layers.Conv3D(128, 3, activation='relu', padding='same')(conv5)
# Decoder
up6 = keras.layers.UpSampling3D(size=(2, 2, 2))(conv5)
up6 = keras.layers.concatenate([up6, conv4], axis=4)
conv6 = keras.layers.Conv3D(64, 3, activation='relu', padding='same')(up6)
conv6 = keras.layers.Conv3D(64, 3, activation='relu', padding='same')(conv6)
up7 = keras.layers.Conv3DTranspose(4, (2, 2, 2), strides=(2, 2, 2), padding='same', output_shape=(None, None, None, None, 4))(conv6)
conv3_upsampled = keras.layers.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv3)
up7 = keras.layers.concatenate([up7, conv3_upsampled], axis=4)
conv7 = keras.layers.Conv3D(32, 3, activation='relu', padding='same')(up7)
conv7 = keras.layers.Conv3D(32, 3, activation='relu', padding='same')(conv7)
up8 = keras.layers.UpSampling3D(size=(2, 2, 2))(conv7)
up8 = keras.layers.concatenate([up8, conv2], axis=4)
conv8 = keras.layers.Conv3D(16, 3, activation='relu', padding='same')(up8)
conv8 = keras.layers.Conv3D(16, 3, activation='relu', padding='same')(conv8)
up9 = keras.layers.UpSampling3D(size=(2, 2, 2))(conv8)
up9 = keras.layers.concatenate([up9, conv1], axis=4)
conv9 = keras.layers.Conv3D(8, 3, activation='relu', padding='same')(up9)
conv9 = keras.layers.Conv3D(8, 3, activation='relu', padding='same')(conv9)

outputs = keras.layers.Conv3D(1, 1, activation='sigmoid')(conv9)

# Create the model
model = keras.models.Model(inputs=[inputs], outputs=[outputs])
model.summary()

# Example usage:
input_shape = (4, 256, 256, 256)
model = keras.models.Sequential()
model.summary()

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.binary_crossentropy,
    metrics=[keras.metrics.binary_accuracy],
)

# Train the model
history = model.fit(train_mri, train_seg, batch_size=1, epochs=50, validation_data=(val_mri, val_seg))