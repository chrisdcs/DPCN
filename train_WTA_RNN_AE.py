import utils.compare as compare
from utils.datasets import mario_loader

import torch
import torch.nn as nn

import os

# set seeds
torch.manual_seed(0)

dataset = 'coil100'
save_dir = "runs/WTA_AE/" + dataset
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Load your data and preprocess it if needed
data = mario_loader("data/coil100_video_train.npy")
data_loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Define the architecture of your autoencoder
autoencoder = compare.WTA_RNN_AE(32*32*3, 10)
autoencoder.to(device)
# Set up your training parameters
num_epochs = 50
learning_rate = 1e-3


# Define your loss function
criterion = nn.MSELoss()

# Define your optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        loss_list = []
        batch = batch.to(torch.float32).to(device)
        loss = 0
        for i in range(batch.shape[-1]-1):
            img_batch = batch[:, :, :, :, i].clone()
            next_img_batch = batch[:, :, :, :, i+1].clone()
            if i == 0: h = None
            img_batch = img_batch.reshape(-1, 32*32*3)#.to(device)
            next_img_batch = next_img_batch.reshape(-1, 32*32*3)#.to(device)
            # Forward pass
            recon, pred, h = autoencoder(img_batch, h)
            
            loss += criterion(recon, img_batch) + 0.1 * criterion(pred, next_img_batch)
            #loss_list.append(loss.item())
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {sum(loss_list)/len(loss_list)}")



# Save the trained autoencoder
torch.save(autoencoder.state_dict(), os.path.join(save_dir,"trained_autoencoder.pth"))