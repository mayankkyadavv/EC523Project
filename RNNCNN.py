# -*- coding: utf-8 -*-
"""Copy of Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eIWJdGveHxw4SjCkXZ_ktMncYxlLQ38d

Step 1: Lets gather our imports
"""

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
drive.mount('/content/drive',force_remount=True)

!wget -O hmdb51.zip http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

!apt-get install unrar

!unrar x "/content/hmdb51.zip" "/content/drive/MyDrive/EC523/EC523Project/"

!unrar x "/content/drive/MyDrive/EC523/EC523Project/*.rar" "/content/drive/MyDrive/EC523/EC523Project/videos"

import cv2
import os

def extract_consecutive_frames(video_path, start_frame, frame_count=30, output_folder='/content/drive/MyDrive/EC523/EC523Project/extracted_frames/April_09_brush_hair_goo_0'):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Position the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Extract consecutive frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            # Save the frame as an image file
            frame_name = os.path.join(output_folder, f"frame_{start_frame + i}.png")
            cv2.imwrite(frame_name, frame)
            print(f"Extracted frame {start_frame + i} saved as '{frame_name}'")
        else:
            print(f"Error: Could not read frame {start_frame + i}")
            break

    # Release the video capture object
    cap.release()

# # Example usage
# extract_consecutive_frames('/content/drive/MyDrive/EC523/EC523Project/videos/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi', 20)

def frames_to_video(frames, output_path, fps):
    height, width, layers = frames[0].shape
    size = (width, height)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()

def crop_video(input_path, output_path, crop_width=240, crop_height=240):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (crop_width, crop_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        start_x = (frame.shape[1] - crop_width) // 2
        cropped_frame = frame[:crop_height, start_x:start_x+crop_width]

        out.write(cropped_frame)

    cap.release()
    out.release()

# Usage
# input_video_path = '/content/drive/MyDrive/EC523/EC523Project/videos/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi'
# output_video_path = '/path/to/output_cropped_video.avi'
# crop_video(input_video_path, output_video_path)

def load_frames_from_directory(directory_path):
    frames = []
    for file in sorted(os.listdir(directory_path)):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
            file_path = os.path.join(directory_path, file)
            frame = cv2.imread(file_path)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Warning: Unable to load {file_path}")
    return frames

import cv2

def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return width, height

video_path = '/content/drive/MyDrive/EC523/EC523Project/videos/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi'
width, height = get_video_dimensions(video_path)

if width and height:
    print(f"Video dimensions: {width}x{height}")
else:
    print("Failed to get video dimensions.")

frames1 = load_frames_from_directory('/content/drive/MyDrive/EC523/EC523Project/extracted_frames/April_09_brush_hair_goo_0')

frames_to_video(frames1, '/content/drive/MyDrive/EC523/EC523Project/concat_vids', 30)

# converts the PIL image to a pytorch tensor containing an LAB image
def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2, 0, 1)))

img_transform = transforms.Compose([
    transforms.Lambda(import_image)
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=img_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=img_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

class ColorNet(nn.Module):
    def __init__(self, d=128):
        super(ColorNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # out: 64 x 8 x 8
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # out: 128 x 4 x 4
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1) # out: 128 x 4 x 4
        self.conv6_bn = nn.BatchNorm2d(128)
        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # out: 64 x 8 x 8
        self.tconv1_bn = nn.BatchNorm2d(64)
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # out: 32 x 16 x 16
        self.tconv2_bn = nn.BatchNorm2d(32)
        self.tconv3 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1) # out: 2 x 32 x 32

    def forward(self, input):
        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.tconv1_bn(self.tconv1(x)))
        x = F.relu(self.tconv2_bn(self.tconv2(x)))
        x = self.tconv3(x)

        return x

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim : int
            Number of channels of input tensor.
        hidden_dim : int
            Number of channels of hidden state.
        kernel_size : (int, int)
            Size of the convolutional kernel.
        bias : bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # for the 4 gates in LSTM
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # out: 32 x 120 x 120
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # out: 64 x 60 x 60
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # out: 128 x 30 x 30
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # out: 128 x 15 x 15
        self.conv4_bn = nn.BatchNorm2d(128)

        # Upsampling layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # out: 64 x 30 x 30
        self.tconv1_bn = nn.BatchNorm2d(64)
        self.tconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # out: 32 x 60 x 60
        self.tconv2_bn = nn.BatchNorm2d(32)
        self.tconv3 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)  # out: 2 x 120 x 120

    def forward(self, input):
        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.tconv1_bn(self.tconv1(x)))
        x = F.relu(self.tconv2_bn(self.tconv2(x)))
        x = self.tconv3(x)
        x = F.interpolate(x, size=(240, 240), mode='bilinear', align_corners=False)

        return x

# class RNN(nn.Module):
#     """
#     Basic RNN block. This represents a single layer of RNN.
#     """
#     def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
#         super().__init__()
#         input_size = 28800
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.i2h = nn.Linear(input_size, hidden_size)
#         self.h2h = nn.Linear(hidden_size, hidden_size)
#         self.h2o = nn.Linear(hidden_size, output_size)

#     def forward(self, x, hidden_state):
#         x = self.i2h(x)
#         hidden_state = self.h2h(hidden_state)
#         hidden_state = torch.tanh(x + hidden_state)
#         out = self.h2o(hidden_state)
#         return out, hidden_state

#     def init_zero_hidden(self, batch_size=1):
#         return torch.zeros(batch_size, self.hidden_size, requires_grad=False).to(device)

class VideoModel(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, kernel_size, bias=True):
        super(VideoModel, self).__init__()
        self.cnn = CNN()
        self.convlstm = ConvLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim,
                                     kernel_size=kernel_size, bias=bias)
        self.decoder = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)  # Assuming num_classes is the output channel size

    def forward(self, video_frames):
        batch_size, timesteps, C, H, W = video_frames.size()
        hidden_state = self.convlstm.init_hidden(batch_size, (H, W))

        outputs = []
        for t in range(timesteps):
            frame = video_frames[:, t, :, :, :]
            frame_features = self.cnn(frame)
            h, c = self.convlstm(frame_features, hidden_state)
            hidden_state = (h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        outputs = self.decoder(outputs)  # Decode to the desired number of output channels
        return outputs

# class VideoModel(nn.Module):
#     """
#     Complete model for processing video frames.
#     """
#     def __init__(self):
#         super().__init__()
#         self.cnn = CNN()  # CNN architecture should be defined in the CNN class
#         # You need to calculate the correct output size of the CNN
#         # For example, if CNN outputs a (240/8) * (240/8) * 64 feature map:
#         cnn_output_size = 64 * (240 // 8) * (240 // 8)
#         self.rnn = RNN(cnn_output_size, hidden_size=256, output_size=256)

#     def forward(self, video_frames):
#         batch_size, timesteps, C, H, W = video_frames.size()
#         hidden_state = self.rnn.init_zero_hidden(batch_size).to(device)  # Ensure hidden state is on the right device
#         outputs = []

#         for t in range(timesteps):
#             frame = video_frames[:, t, :, :, :]
#             frame_features = self.cnn(frame)
#             frame_features = frame_features.view(batch_size, -1)  # Flatten the CNN output
#             out, hidden_state = self.rnn(frame_features, hidden_state)
#             outputs.append(out)

#         return torch.stack(outputs, dim=1)

model = ColorNet()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

epochs = 25
model.train()

train_loss_avg = []

print('Training ...')
for epoch in range(epochs):
    train_loss_avg.append(0)
    num_batches = 0

    for lab_batch, _ in train_loader:

        lab_batch = lab_batch.to(device)

        # apply the color net to the luminance component of the Lab images
        # to get the color (ab) components
        predicted_ab_batch = model(lab_batch[:, 0:1, :, :])

        # loss is the L2 error to the actual color (ab) components
        loss = F.mse_loss(predicted_ab_batch, lab_batch[:, 1:3, :, :])

        # backpropagation
        optimizer.zero_grad()
        loss.backward()

        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()

        train_loss_avg[-1] += loss.item()
        num_batches += 1

    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, epochs, train_loss_avg[-1]))

import matplotlib.pyplot as plt
plt.ion()

fig = plt.figure(figsize=(15, 5))
plt.plot(train_loss_avg)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

model.eval()

test_loss_avg, num_batches = 0, 0
for lab_batch, _ in test_loader:

    with torch.no_grad():

        lab_batch = lab_batch.to(device)

        # apply the color net to the luminance component of the Lab images
        # to get the color (ab) components
        predicted_ab_batch = model(lab_batch[:, 0:1, :, :])

        # loss is the L2 error to the actual color (ab) components
        loss = F.mse_loss(predicted_ab_batch, lab_batch[:, 1:3, :, :])

        test_loss_avg += loss.item()
        num_batches += 1

test_loss_avg /= num_batches
print('Test loss: %f' % (test_loss_avg))

import numpy as np
from skimage import color, io

import matplotlib.pyplot as plt
plt.ion()

import torchvision.utils

with torch.no_grad():

    # pick a random subset of images from the test set
    image_inds = np.random.choice(len(test_dataset), 25, replace=False)
    lab_batch = torch.stack([test_dataset[i][0] for i in image_inds])
    lab_batch = lab_batch.to(device)

    # predict colors (ab channels)
    predicted_ab_batch = model(lab_batch[:, 0:1, :, :])
    predicted_lab_batch = torch.cat([lab_batch[:, 0:1, :, :], predicted_ab_batch], dim=1)

    lab_batch = lab_batch.cpu()
    predicted_lab_batch = predicted_lab_batch.cpu()

    # convert to rgb
    rgb_batch = []
    predicted_rgb_batch = []
    for i in range(lab_batch.size(0)):
        rgb_img = color.lab2rgb(np.transpose(lab_batch[i, :, :, :].numpy().astype('float64'), (1, 2, 0)))
        rgb_batch.append(torch.FloatTensor(np.transpose(rgb_img, (2, 0, 1))))
        predicted_rgb_img = color.lab2rgb(np.transpose(predicted_lab_batch[i, :, :, :].numpy().astype('float64'), (1, 2, 0)))
        predicted_rgb_batch.append(torch.FloatTensor(np.transpose(predicted_rgb_img, (2, 0, 1))))

    # plot images
    fig, ax = plt.subplots(figsize=(15, 15), nrows=1, ncols=2)
    ax[0].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(predicted_rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[0].title.set_text('re-colored')
    ax[1].imshow(np.transpose(torchvision.utils.make_grid(torch.stack(rgb_batch), nrow=5).numpy(), (1, 2, 0)))
    ax[1].title.set_text('original')
    plt.show()

torch.save(model.state_dict(), 'BW_img_colorizationV1.pth')

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

input_video_path = '/content/drive/MyDrive/EC523/EC523Project/videos/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi'
output_video_path = '/content/drive/MyDrive/EC523/EC523Project/cropped_vid/hair_cropped_video.avi'
crop_video(input_video_path, output_video_path)


# frames = extract_frames(video_path)
# grayscale_frames = convert_to_grayscale(frames)
# processed_frames = preprocess_frames(grayscale_frames)

def convert_to_grayscale(frames):
    grayscale_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(gray)
    return grayscale_frames

video_path = "/content/drive/MyDrive/EC523/EC523Project/cropped_vid/hair_cropped_video.avi"
frames = extract_frames(video_path)
grayscale_frames = convert_to_grayscale(frames)

print(grayscale_frames)

from google.colab.patches import cv2_imshow
def display_sample_images(frames, sample_size=5):
    for i in range(min(sample_size, len(frames))):
        cv2_imshow(frames[i])
        cv2.waitKey(0)  # Wait for a key press to show the next image
    cv2.destroyAllWindows()

display_sample_images(grayscale_frames)

def colorize_video(model, frames):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        colorized_frames = model(frames.to(device))  # Ensure frames are on the same device as the model
    return colorized_frames

def preprocess_frames(frames):
    processed_frames = []
    for frame in frames:
        normalized_frame = frame / 255.0  # Normalize
        tensor_frame = torch.tensor(normalized_frame, dtype=torch.float32)
        tensor_frame = tensor_frame.unsqueeze(0)  # Add channel dimension if grayscale
        processed_frames.append(tensor_frame)

    # Stack all frames into a single tensor: shape (timesteps, C, H, W)
    tensor_frames = torch.stack(processed_frames)

    # Add a batch dimension: shape (1, timesteps, C, H, W)
    tensor_frames = tensor_frames.unsqueeze(0)
    return tensor_frames

# Assuming the output of your CNN is 128 channels
cnn_output_channels = 128  # This should match the output channels of your CNN

# Number of output classes - For colorized video, this might be 3 (for RGB channels)
num_classes = 3

# Hidden dimension for ConvLSTM - This is a hyperparameter you can tune
hidden_dim = 128  # Example value, you can adjust this

# Kernel size for ConvLSTM - Typically a small odd number like (3, 3) or (5, 5)
kernel_size = (3, 3)

# Initialize the VideoModel
model = VideoModel(num_classes=num_classes, input_dim=cnn_output_channels,
                   hidden_dim=hidden_dim, kernel_size=kernel_size)

# model = VideoModel()
model.to(device)

input = preprocess_frames(grayscale_frames)
print(input.shape)
input = input.to(device)
colored_frames = colorize_video(model, input)

print(colored_frames.shape)

import matplotlib.pyplot as plt

# Assuming the RNN output is [1, 408, 256]
num_frames = colored_frames.size(1)

for i in range(min(num_frames, 5)):  # Visualize first 5 frames
    # Reshape 256-dimensional vector into a 16x16 image
    frame = colored_frames[0, i].view(16, 16).cpu().numpy()

    plt.imshow(frame, cmap='gray')  # Grayscale visualization
    plt.title(f'Frame {i}')
    plt.axis('off')
    plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import color
# import torchvision.utils

# plt.ion()

# with torch.no_grad():
#     # Assuming colorized_frames is a tensor of shape [num_frames, C, H, W]
#     # where C = 3 for RGB or 1 for grayscale, H and W are height and width of the frame

#     # Select 5 random frames for display
#     num_frames = colored_frames.size(0)
#     frame_inds = np.random.choice(num_frames, 5, replace=True)
#     selected_frames = colored_frames[frame_inds]

#     # Convert to RGB if necessary (e.g., if frames are in LAB or grayscale)
#     # If the frames are already in RGB, you can skip this conversion
#     rgb_frames = []
#     for i in range(len(selected_frames)):
#         # Example conversion, adjust based on your color space
#         # rgb_img = color.lab2rgb(np.transpose(selected_frames[i].numpy(), (1, 2, 0)))
#         rgb_img = np.transpose(selected_frames[i].numpy(), (1, 2, 0))  # If already in RGB
#         rgb_frames.append(torch.FloatTensor(rgb_img))

#     # Plot the images
#     fig, ax = plt.subplots(figsize=(15, 5))
#     ax.imshow(np.transpose(torchvision.utils.make_grid(torch.stack(rgb_frames), nrow=5).numpy(), (1, 2, 0)))
#     ax.title.set_text('Colorized Frames')
#     ax.axis('off')  # Disable axis for better visualization
#     plt.show()

