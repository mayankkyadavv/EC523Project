!pip install pytorch-msssim

!pip install av

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from skimage import color

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("using cuda")
else:
  device = torch.device("cpu")
  print("using cpu")

from google.colab import drive
drive.mount('/content/drive')

#import os

#image_dataset_path = '/content/drive/MyDrive/EC523Project/coco'
#os.makedirs(image_dataset_path, exist_ok=True)

#!wget http://images.cocodataset.org/zips/val2017.zip -P {image_dataset_path}

#!wget http://images.cocodataset.org/zips/train2017.zip -P {image_dataset_path}

#!unzip -q /content/drive/MyDrive/EC523Project/coco/val2017.zip -d /content/drive/MyDrive/EC523Project/coco/val
#!unzip -q /content/drive/MyDrive/EC523Project/coco/train2017.zip -d /content/drive/MyDrive/EC523Project/coco/train

#import cv2
#import os

#def extract_frames(video_path, frame_dir, frame_rate=2):
#    video = cv2.VideoCapture(video_path)
#    frame_count = 0
#
 #   while True:
  #      success, image = video.read()
   #     if not success:
    #        break
#
 #
  #      if frame_count % (30 // frame_rate) == 0:
   #         frame_path = os.path.join(frame_dir, f'frame{frame_count}.jpg')
    #        cv2.imwrite(frame_path, image)
#
 #       frame_count += 1
#
 ##
#video_root_dir = '/content/drive/MyDrive/EC523Project/video_extracted/'
#frames_root_dir = os.path.join(video_root_dir, 'frames')

#if not os.path.exists(frames_root_dir):
#    os.makedirs(frames_root_dir)
#
#for category in os.listdir(video_root_dir):
#    category_path = os.path.join(video_root_dir, category)
#    if os.path.isdir(category_path):
#        for i, video in enumerate(os.listdir(category_path)):
#            video_path = os.path.join(category_path, video)
#            frame_dir = os.path.join(frames_root_dir, category, f'video{i+1}_frames')
#            os.makedirs(frame_dir, exist_ok=True)


#            extract_frames(video_path, frame_dir, frame_rate=2)

#print("Frame extraction complete.")

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image
from skimage import color
from PIL import Image

class MixedDataset(Dataset):
    def __init__(self, coco_dir, frame_dir, transform=None, subset_size=None, num_videos=None):
        self.coco_dir = coco_dir
        self.frame_dir = frame_dir
        self.transform = transform

        #load COCO images
        coco_images = os.listdir(coco_dir)
        if subset_size:
            coco_images = random.sample(coco_images, min(subset_size, len(coco_images)))
        self.coco_images = [os.path.join(coco_dir, img) for img in coco_images]

        #load video frame paths
        all_video_folders = []
        for category in os.listdir(frame_dir):
            category_path = os.path.join(frame_dir, category)
            if os.path.isdir(category_path):
                video_folders = [os.path.join(category_path, video_folder) for video_folder in os.listdir(category_path)]
                all_video_folders.extend(video_folders)

        #select a subset of videos if num_videos is specified
        if num_videos and num_videos < len(all_video_folders):
            self.video_frame_folders = random.sample(all_video_folders, num_videos)
        else:
            self.video_frame_folders = all_video_folders

        self.dataset = self.coco_images + self.video_frame_folders
        random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        file_path = self.dataset[idx]
        is_video = file_path.startswith(self.frame_dir)

        if is_video:
            frames = sorted(os.listdir(file_path))
            transformed_sequence = []
            for frame in frames:
                frame_path = os.path.join(file_path, frame)
                image = Image.open(frame_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                transformed_sequence.append(image)
            return transformed_sequence, True
        else:
            image = Image.open(file_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return [image], False

#function to convert image to LAB color space
def import_image(img):
    lab_image = color.rgb2lab(np.array(img))
    lab_image[0] = (lab_image[0] - 50) / 50  # Normalize L channel
    lab_image[1] = lab_image[1] / 128        # Normalize A and B channels
    lab_image[2] = lab_image[2] / 128
    return torch.FloatTensor(np.transpose(lab_image, (2, 0, 1)))


random.seed(42)

#transformation
uniform_size = (256, 256)
img_transform = transforms.Compose([
    transforms.Resize(uniform_size),
    transforms.Lambda(import_image),
])

#paths to your datasets
train_path = '/content/drive/MyDrive/EC523Project/coco/train/train2017/'
val_path = '/content/drive/MyDrive/EC523Project/coco/val/val2017/'
video_dir = '/content/drive/MyDrive/EC523Project/video_extracted/frames/'

#create custom datasets
num_videos = 100
train_dataset = MixedDataset(coco_dir=train_path, frame_dir=video_dir, transform=img_transform, subset_size=6000, num_videos=num_videos)
val_dataset = MixedDataset(coco_dir=val_path, frame_dir=video_dir, transform=img_transform, subset_size=700, num_videos=num_videos)

def custom_collate_fn(batch):
    images = []
    is_video_flags = []
    for item in batch:
        sequence, is_video = item
        images.extend(sequence)
        is_video_flags.extend([is_video] * len(sequence))
    images_tensor = torch.stack(images)
    is_video_flags_tensor = torch.tensor(is_video_flags, dtype=torch.bool)
    return images_tensor, is_video_flags_tensor

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)

import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => LeakyReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW. Handle padding for odd dimensions during concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()

        #encoder (downsampling path)
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        #LSTM layer for video frames
        self.lstm = nn.LSTM(input_size=512, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 512)

        #decoder (upsampling path)
        self.up1 = up(1024, 256, bilinear)
        self.up2 = up(512, 128, bilinear)
        self.up3 = up(256, 64, bilinear)
        self.up4 = up(128, 64, bilinear)
        self.outc = outconv(64, n_classes)

    def forward(self, x, is_video_flags):
        #encoding
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #reshape for LSTM: assuming x5 is of shape [batch, channels, height, width]
        b, c, h, w = x5.shape
        x5_flattened = x5.view(b, c, -1).permute(0, 2, 1)  # Reshape to [batch, seq_len, features]
        lstm_output, _ = self.lstm(x5_flattened)
        lstm_output = self.fc(lstm_output)
        lstm_output = lstm_output.permute(0, 2, 1).view(b, c, h, w)  # Reshape back

        x5 = torch.where(is_video_flags.unsqueeze(1).unsqueeze(2).unsqueeze(3), lstm_output, x5)

        #decoding
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

model = UNet(n_channels=1, n_classes=2)
model.to(device)
#print(model)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

import matplotlib.pyplot as plt
import torch.optim as optim
#from pytorch_msssim import ssim

import gc
import torch

model.train()

epochs = 5
train_loss_avg = []

print('Training ...')
for epoch in range(epochs):
    train_loss_avg.append(0)
    num_batches = 0

    for batch_idx, (images, is_video_flags) in enumerate(train_loader):
        images = images.to(device)
        is_video_flags = is_video_flags.to(device)

        #split images into L and AB components for the LAB color space
        L = images[:, 0:1, :, :]
        AB = images[:, 1:3, :, :]


        predicted_ab_batch = model(L, is_video_flags)


        loss = criterion(predicted_ab_batch, AB)


        optimizer.zero_grad()
        loss.backward()


        optimizer.step()

        train_loss_avg[-1] += loss.item()
        num_batches += 1

        #clear memory
        del images, is_video_flags, L, AB, predicted_ab_batch, loss
        gc.collect()

        print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(train_loader)}, Batch Loss: {train_loss_avg[-1] / num_batches:.4f}')

    train_loss_avg[-1] /= num_batches
    print(f'Epoch [{epoch+1}/{epochs}] Average Loss: {train_loss_avg[-1]:.4f}')

# Plotting the training loss
plt.figure(figsize=(15, 5))
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

torch.save(model.state_dict(), '/content/drive/MyDrive/EC523Project/train1.pth')

model = UNet(n_channels=1, n_classes=2)

model.load_state_dict(torch.load('/content/drive/MyDrive/EC523Project/statepath.pth'))

model = model.to(device)

from pytorch_msssim import ssim
import matplotlib.pyplot as plt
from skimage import color
import numpy as np

def lab_to_rgb(L, AB):
    LAB = torch.cat([L, AB], dim=1).data.cpu().numpy()
    RGBs = []
    for lab in LAB:
        lab = np.transpose(lab, (1, 2, 0))
        rgb = color.lab2rgb(lab)
        RGBs.append(rgb)
    return RGBs

model.eval()

total_ssim_loss = 0.0
count = 0

original_images = []
recolorized_images = []

with torch.no_grad():
    for lab_batch, is_video_flags in val_loader:
        for i in range(len(lab_batch)):
            lab_image = lab_batch[i]
            is_video_flag = is_video_flags[i]


            if lab_image.dim() == 3:
                lab_image = lab_image.unsqueeze(0)

            L = lab_image[:, 0:1, :, :].to(device)
            AB = lab_image[:, 1:3, :, :].to(device)
            is_video_flag = is_video_flag.unsqueeze(0).to(device)


            predicted_AB = model(L, is_video_flag)

            ssim_val = ssim(predicted_AB, AB, data_range=1, size_average=True)
            total_ssim_loss += ssim_val.item()
            count += 1


            if i < 5:
                original = lab_to_rgb(L, AB)
                recolorized = lab_to_rgb(L, predicted_AB)
                original_images.extend(original)
                recolorized_images.extend(recolorized)

average_ssim_loss = total_ssim_loss / count
print(f'Average SSIM Loss on Validation Set: {average_ssim_loss:.4f}')

# Displaying original and recolorized images
plt.figure(figsize=(100, 100))
for i in range(len(original_images)):
    # Original
    plt.subplot(2, len(original_images), i + 1)
    plt.imshow(original_images[i])
    plt.title('Original')
    plt.axis('off')

    # Recolorized
    plt.subplot(2, len(original_images), i + len(original_images) + 1)
    plt.imshow(recolorized_images[i])
    plt.title('Recolorized')
    plt.axis('off')

plt.show()

# Displaying original and recolorized images in a 5x2 grid
num_images_to_show = 20
plt.figure(figsize=(100, 100))

for i in range(num_images_to_show):
    # Original
    plt.subplot(num_images_to_show, 2, 2*i + 1)
    plt.imshow(original_images[i])
    plt.title('Original')
    plt.axis('off')

    # Recolorized
    plt.subplot(num_images_to_show, 2, 2*i + 2)
    plt.imshow(recolorized_images[i])
    plt.title('Recolorized')
    plt.axis('off')

plt.tight_layout()
plt.show()
