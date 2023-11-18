import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from google.colab import drive
drive.mount('/content/drive')


#define transformations, greyscale images, normalize

#input transformation
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

#expected output transformation
transform_target = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




#class to handle data transformations 
class CIFAR10Colorization(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)

        # Input (grayscale) and target (color)
        input_img = transform(img)
        target_img = transform_target(img)

        return input_img, target_img
    
#STILL NEEDED: make function for video transformation

trainset = CIFAR10Colorization(root='./data', train=True, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = CIFAR10Colorization(root='./data', train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


#class for NN
class ColorizationNet(nn.Module):

    #CODE TO COMPLETE: RNN integration and Video Data integration
    def __init__(self):
        super(ColorizationNet, self).__init__()
        #downsampling edit
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        #upsampling edit
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upsample3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)  #output 3 channels (RGB)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.upsample1(x))
        x = F.relu(self.upsample2(x))
        x = torch.sigmoid(self.upsample3(x)) 
        return x




# Training model 
net = ColorizationNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 5

print("Starting Training...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data

        #zero grad
        optimizer.zero_grad()

        #compute
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # print statistics for every 100 mini-batches
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')



def imshow(img):
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




# test data
dataiter = iter(testloader)
images, labels = dataiter.next()

# first image in the batch
input_image = images[0]
true_image = labels[0]

# grayscale input image
print('Grayscale Input Image:')
imshow(torchvision.utils.make_grid(input_image))

# Colorize the image 
output = net(input_image.unsqueeze(0))

# colorized output image
print('Colorized Output Image:')
imshow(torchvision.utils.make_grid(output.detach().squeeze(0)))

# comparison
print('True Color Image:')
imshow(torchvision.utils.make_grid(true_image))






# Calculating MSE and SSIM 
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_mse(target, output):
    return ((target - output) ** 2).mean()

def calculate_ssim(target, output, data_range=1):
    # Converting tensors to numpy arrays
    target_np = target.squeeze().cpu().numpy()
    output_np = output.squeeze().cpu().detach().numpy()
    
    # Calculating SSIM
    return ssim(target_np, output_np, data_range=data_range, multichannel=True)

# Example usage in the test loop
total_mse = 0.0
total_ssim = 0.0
total_batches = 0

for i, data in enumerate(testloader, 0):
    inputs, targets = data
    outputs = net(inputs)

    total_mse += calculate_mse(targets, outputs)
    total_ssim += calculate_ssim(targets, outputs)
    total_batches += 1

average_mse = total_mse / total_batches
average_ssim = total_ssim / total_batches

print(f"Average MSE on Test Set: {average_mse}")
print(f"Average SSIM on Test Set: {average_ssim}")
