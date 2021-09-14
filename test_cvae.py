import torch
import numpy as np
import matplotlib.pyplot as plt
from cvae import CVAutoencoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load gpu-created-data: add map_location=device in case loaded when device="cpu"
PATH = './cvae_net.pt'
net_state_dict = torch.load(PATH, map_location=device)

model = CVAutoencoder()
model.load_state_dict(net_state_dict)
model.to(device)


c = torch.eye(10, 10).to(device)
latent_sample = torch.randn(10, model.latent_dim).to(device)
sample = model.cvdecoder(latent_sample, c).detach().cpu()

num = np.random.randint(0,9)
plt.imshow(sample[num].view(28, 28))
plt.title(num)
plt.axis('off')
plt.show()
