import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        """
        Initializes the UNet model.

        The UNet model is a series of convolutional layers with ReLU activation, 
        followed by max pooling and upsampling. The input to the model is a 3D 
        image, and the output is a 3D image with the same shape as the input. 
        The layers are as follows:

        1. Conv3d: 1 -> 64
        2. Conv3d: 64 -> 64
        3. MaxPool3d: 64 -> 64
        4. Conv3d: 64 -> 128
        5. Conv3d: 128 -> 128
        6. MaxPool3d: 128 -> 128
        7. Conv3d: 128 -> 256
        8. Conv3d: 256 -> 256
        9. MaxPool3d: 256 -> 256
        10. Conv3d: 256 -> 512
        11. Conv3d: 512 -> 512
        12. MaxPool3d: 512 -> 512
        13. Conv3d: 512 -> 1024
        14. Conv3d: 1024 -> 1024
        15. ConvTranspose3d: 1024 -> 512
        16. Conv3d: 512 -> 256
        17. ConvTranspose3d: 256 -> 128
        18. Conv3d: 128 -> 64
        19. ConvTranspose3d: 64 -> 3

        """
        super(UNet, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv3d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv3d(128, 256, kernel_size=3)
        self.conv6 = nn.Conv3d(256, 256, kernel_size=3)
        self.conv7 = nn.Conv3d(256, 512, kernel_size=3)
        self.conv8 = nn.Conv3d(512, 512, kernel_size=3)
        self.conv9 = nn.Conv3d(512, 1024, kernel_size=3)
        self.conv10 = nn.Conv3d(1024, 1024, kernel_size=3)
        self.conv11 = nn.Conv3d(1024, 512, kernel_size=2)
        self.conv12 = nn.Conv3d(512, 256, kernel_size=2)
        self.conv13 = nn.Conv3d(256, 128, kernel_size=2)
        self.conv14 = nn.Conv3d(128, 64, kernel_size=2)
        self.conv15 = nn.Conv3d(64, 3, kernel_size=2)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x: The input 3D image

        Returns:
            The output of the network, which is a 3D image with the same shape as the input.
        """
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv2(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv3(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv4(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv5(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv6(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv7(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv8(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv9(x), 2))
        x = nn.functional.relu(nn.functional.max_pool3d(self.conv10(x), 2))
        x = nn.functional.relu(self.conv11(x))
        x = nn.functional.relu(self.conv12(x))
        x = nn.functional.relu(self.conv13(x))
        x = nn.functional.relu(self.conv14(x))
        x = self.conv15(x)
        return x