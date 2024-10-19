from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch
import matplotlib
from torchvision.utils import draw_bounding_boxes
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# Import maskrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn


train_dir = "/home/jonas/Documents/Learning/python_dev/image_recognition/data/train"


anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128),),
    aspect_ratios=((0.5, 1.0, 2.0),),
)

roi_pooler = MultiScaleRoIAlign(
    featmap_names=["0"],
    output_size=7,
    sampling_ratio=2,
)


class BinaryCNN(nn.Module):
    def __init__(self):
        super(BinaryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 112 * 112 * 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.fc1(self.flatten(x))
        x = self.sigmoid(x)
        return x


class MultiClassImageClassifier(nn.Module):

    # Define the init method
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Create a fully connected layer
        self.fc = nn.Linear(16 * 32 * 32, num_classes)

        # Create an activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.softmax(x)
        return x


class BinaryImageClassifier(nn.Module):
    def __init__(self):
        super(BinaryImageClassifier, self).__init__()

        # Create a convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Create a fully connected layer
        self.fc = nn.Linear(16 * 32 * 32, 1)

        # Create an activation function
        self.sigmoid = nn.Sigmoid()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)


class BinaryImageClassification(nn.Module):
    def __init__(self):
        super(BinaryImageClassification, self).__init__()
        # Create a convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # Pass inputs through the convolutional block
        x = self.conv_block(x)
        return x


class ObjectDetectorCNN(nn.Module):
    def __init__(self):
        super(ObjectDetectorCNN, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(vgg.features.children()))
        input_features = nn.Sequential(*list(vgg.classifier.children()))[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )
        self.box_regressor = nn.Sequential(
            nn.Linear(input_features, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

    def forward(self, x):
        features = self.backbone(x)
        bboxes = self.regressor(features)
        classes = self.classifier(features)
        return bboxes, classes


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block
        self.enc3 = self.conv_block
        self.enc4 = self.conv_block

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Define the decoder blocks
        self.dec1 = self.conv_block
        self.dec2 = self.conv_block
        self.dec3 = self.conv_block

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        x = self.upconv3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.dec1(x)

        x = self.upconv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        # Define the last decoder block with skip connections
        x = self.upconv1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)

        return self.out(x)


# class Generator(nn.Module):
#     def __init__(self, in_dim, out_dim):
#         super(Generator, self).__init__()
#         # Define generator block
#         self.generator = nn.Sequential(
#             gen_block(in_dim, 256),
#             gen_block(256, 512),
#             gen_block(512, 1024),
#             # Add linear layer
#             nn.Linear(1024, out_dim),
#             # Add activation
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         # Pass input through generator
#         return self.generator(x)


# class Discriminator(nn.Module):
#     def __init__(self, im_dim):
#         super(Discriminator, self).__init__()
#         self.disc = nn.Sequential(
#             disc_block(im_dim, 1024),
#             disc_block(1024, 512),
#             # Define last discriminator block
#             disc_block(512, 256),
#             # Add a linear layer
#             nn.Linear(256, 1),
#         )

#     def forward(self, x):
#         # Define the forward method
#         return self.disc(x)

# class DCGenerator(nn.Module):
#     def __init__(self, in_dim, kernel_size=4, stride=2):
#         super(DCGenerator, self).__init__()
#         self.in_dim = in_dim
#         self.gen = nn.Sequential(
#             dc_gen_block(in_dim, 1024, kernel_size, stride),
#             dc_gen_block(1024, 512, kernel_size, stride),
#             # Add last generator block
#             dc_gen_block(512, 256, kernel_size, stride),
#             # Add transposed convolution
#             nn.ConvTranspose2d(256, 3, kernel_size, stride=stride),
#             # Add tanh activation
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = x.view(len(x), self.in_dim, 1, 1)
#         return self.gen(x)

# class DCDiscriminator(nn.Module):
#     def __init__(self, kernel_size=4, stride=2):
#         super(DCDiscriminator, self).__init__()
#         self.disc = nn.Sequential(
#           	# Add first discriminator block
#             dc_disc_block(3, 512, kernel_size, stride),
#             dc_disc_block(512, 1024, kernel_size, stride),
#           	# Add a convolution
#             nn.Conv2d(1024, 1, kernel_size, stride=stride),
#         )

#     def forward(self, x):
#         # Pass input through sequential block
#         x = x.view(len(x), self.in_dim, 1, 1)
#         return x.view(len(x), -1)


def gen_loss(gen, disc, criterion, num_images, z_dim):
    # Define random noise
    noise = torch.randn(num_images, z_dim)
    # Generate fake image
    fake = gen(noise)
    # Get discriminator's prediction on the fake image
    disc_pred = disc(fake)
    # Compute generator loss
    criterion = nn.BCEWithLogitsLoss()
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss


def disc_loss(gen, disc, real, num_images, z_dim):
    criterion = nn.BCEWithLogitsLoss()
    noise = torch.randn(num_images, z_dim)
    fake = gen(noise)
    # Get discriminator's predictions for fake images
    disc_pred_fake = disc(fake)
    # Calculate the fake loss component
    fake_loss = criterion(disc_pred_fake, torch.ones_like(disc_pred_fake))
    # Get discriminator's predictions for real images
    disc_pred_real = disc(real)
    # Calculate the real loss component
    real_loss = criterion(disc_pred_real, torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2
    return disc_loss


# for epoch in range(1):
#     for real in dataloader:
#         cur_batch_size = len(real)

#         disc_opt.zero_grad()
#         # Calculate discriminator loss
#         disc_loss = disc_loss(gen, disc, real, cur_batch_size, z_dim=16)
#         # Compute gradients
#         disc_loss.backward()
#         disc_opt.step()

#         gen_opt.zero_grad()
#         # Calculate generator loss
#         gen_loss = gen_loss(gen, disc, cur_batch_size, z_dim=16)
#         # Compute generator gradients
#         gen_loss.backward()
#         gen_opt.step()

#         print(f"Generator loss: {gen_loss}")
#         print(f"Discriminator loss: {disc_loss}")
#         break


def main():
    # Collecting the classes from the folder
    train_dataset = ImageFolder(root=train_dir, transform=transforms.ToTensor())
    classes = train_dataset.classes

    print(classes)
    print(train_dataset.class_to_idx)

    # Getting number of channels from an image
    image = Image.open(
        "/home/jonas/Documents/Learning/python_dev/image_recognition/data/train/dog/dog.jpg"
    )
    num_channels = functional.get_image_num_channels(image)
    print("Number of channels: ", num_channels)

    # Adding convolutional layers
    conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
    model = Net()
    model.add_module("conv2", conv2)
    print(model)

    # Generating new input image
    image = Image.open(
        "/home/jonas/Documents/Learning/python_dev/image_recognition/data/train/cat/cat.jpg"
    )
    image_tensor = transforms.ToTensor()(image)
    image_reshaped = image_tensor.permute(1, 2, 0).unsqueeze(0)
    plt.imshow(image_reshaped.squeeze())
    plt.savefig("image.png")

    # Generating a new prediction
    # # Apply preprocessing transforms
    # batch = preprocess(img).unsqueeze(0)

    # # Apply model with softmax layer
    # prediction = model(batch).squeeze(0).softmax(0)

    # # Apply argmax
    # class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(category_name)

    # Drawing the bounding box
    bbox = torch.tensor([30, 10, 200, 150])
    bbox_tensor = torch.tensor(bbox).unsqueeze(0)

    img_bbox = draw_bounding_boxes(image_tensor, bbox_tensor, width=3, colors="red")

    transform = transforms.Compose([transforms.ToPILImage()])
    plt.imshow(transform(img_bbox))
    plt.show()
    plt.savefig("image-box.png")

    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Load an image and convert to a tensor
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)
        print(prediction)

    # Load model
    model = UNet()
    model.eval()

    # Load and transform image
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    # Predict segmentation mask
    with torch.no_grad():
        prediction = model(image_tensor).squeeze(0)

    # Display mask
    plt.imshow(prediction[1, :, :])
    plt.show()


if __name__ == "__main__":
    main()
