import utils.compare as compare
from utils.datasets import image_loader

import torch
import torch.nn as nn

import os

# set seeds
torch.manual_seed(0)

dataset = 'mario'
save_dir = "runs/WTA_AE/" + dataset
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Load your data and preprocess it if needed
data = image_loader("data/mario_video_train.npy")
data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Define the architecture of your autoencoder
autoencoder = compare.WTA_RNN_AE(32*32*3, 10)
autoencoder.to(device)
# Set up your training parameters
num_epochs = 50
batch_size = 32
learning_rate = 1e-3


# Define your loss function
criterion = nn.MSELoss()

# Define your optimizer
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        
        batch = batch.reshape(-1, 32*32*3).to(device)
        # Forward pass
        output = autoencoder(batch)

        # Compute the loss
        loss = criterion(output, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")



# Save the trained autoencoder
torch.save(autoencoder.state_dict(), os.path.join(save_dir,"trained_autoencoder.pth"))