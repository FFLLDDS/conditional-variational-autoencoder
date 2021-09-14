import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from cvae import CVAutoencoder

PATH = 'data'
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LATENT_DIM = 1
EPOCHS = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = torchvision.transforms.ToTensor()

train_data = torchvision.datasets.MNIST(root=PATH, train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transform)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


cvae = CVAutoencoder()
cvae.to(device)

optimizer = torch.optim.Adam(cvae.parameters(), lr=LEARNING_RATE)

# training
print('starting training cvae...')
training_loss = []

for epoch in range(EPOCHS): 
    
    epoch_loss = 0.0 
    for batch in train_dataloader:
        im, label = batch
        im, label = im.to(device), label.to(device)
        y = torch.nn.functional.one_hot(label, num_classes=10)
        output, mu, sigma = cvae(im, y)
        
        loss = cvae.loss_function(im, output, mu, sigma)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()*im.size(0)
    
    epoch_loss = epoch_loss / len(train_dataloader)
    training_loss.append(epoch_loss)
    print(f'epoch {epoch+1} loss: {epoch_loss:.3f}')

print('Finished training.')  
# save the net
PATH = './cvae_net.pt'
torch.save(cvae.state_dict(), PATH)

# print loss against epoch
fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.plot(range(1, EPOCHS + 1), training_loss, 'r.')
plt.xticks(range(1, EPOCHS + 1 ))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

