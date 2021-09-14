import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Parent(nn.Module):
    def __init__(self, latent_dim, input_height, input_width, num_classes, num_channels):
        """Initializes object with attributes described under Args. 

            Args:
                latent_dim (int, optional): Dimension of the latent space representation of the input. Defaults to 1.
                input_height (int, optional): Height of input images in pixels. Defaults to 28.
                input_width (int, optional): Width of input images in pixels. Defaults to 28.
                num_classes (int, optional): Number of classes to condition on. Defaults to 10.
                num_channels (int, optional): Number of input channels. Defaults to 1."""
                
        super().__init__()
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.num_channels = num_channels


class CVEncoder(Parent):
    def __init__(self, latent_dim, input_height, input_width, num_classes, num_channels):
        """Initializes CVEncoder object with the following arguments. 

            Args:
                latent_dim (int, optional): Dimension of the latent space representation of the input. 
                input_height (int, optional): Height of input images in pixels. 
                input_width (int, optional): Width of input images in pixels. 
                num_classes (int, optional): Number of classes to condition on. 
                num_channels (int, optional): Number of input channels."""
        super().__init__(latent_dim, input_height, input_width, num_classes, num_channels)
        
        self.final_height = int(self.input_height / 4)
        self.final_width = int(self.input_width / 4)
        
        self.fc0 = nn.Linear(self.input_height * self.input_width + self.num_classes, self.input_height * self.input_width)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc_mean = nn.Linear(32 * self.final_height * self.final_width, self.latent_dim)
        self.fc_std = nn.Linear(32 * self.final_height * self.final_width, self.latent_dim)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, input, c):
        x = input.view(-1, self.input_height * self.input_width)
        # fuse sample with labels
        x = self.fc0(torch.cat([x, c], 1))
        x = x.view(-1, 1, self.input_height, self.input_width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, 32 * self.final_height * self.final_width)
        mu = self.fc_mean(x)
        sigma = self.fc_std(x)
        return mu, sigma
    
    
class CVDecoder(Parent):
    def __init__(self, latent_dim, input_height, input_width, num_classes, num_channels):
        """Initializes CVDecoder object with the following arguments. 

            Args:
                latent_dim (int, optional): Dimension of the latent space representation of the input. 
                input_height (int, optional): Height of input images in pixels. 
                input_width (int, optional): Width of input images in pixels. 
                num_classes (int, optional): Number of classes to condition on. 
                num_channels (int, optional): Number of input channels."""
        super().__init__(latent_dim, input_height, input_width, num_classes, num_channels)
        
        self.fc0 = nn.Linear(self.latent_dim + 10, 16*7*7)
        self.fc1 = nn.Linear(16*7*7, 32*7*7)
        self.bn_fc = nn.BatchNorm2d(32)
        
        self.t_conv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, padding=0, stride=2)

        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, padding=0, stride=2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, c):
        x = self.fc0(torch.cat([z, c], 1))
        x = self.relu(x)
        x = self.fc1(x)
        x = x.view(-1, 32, 7, 7)
        x = self.bn_fc(x)
        x = self.relu(x)
        
        x = self.t_conv1(x)
        x = self.relu(x)

        x = self.t_conv2(x)
        x = self.sigmoid(x)


        return x

class CVAutoencoder(Parent):
    def __init__(self, latent_dim=1, input_height=28, input_width=28, num_classes=10, num_channels=1, beta=1):
        """Initializes CVAutoencoder object with the following arguments. Instantiates a CVEncoder and CVDecoder object with the following arguments, except beta. 

            Args:
                latent_dim (int, optional): Dimension of the latent space representation of the input. Defaults to 1.
                input_height (int, optional): Height of input images in pixels. Defaults to 28.
                input_width (int, optional): Width of input images in pixels. Defaults to 28.
                num_classes (int, optional): Number of classes to condition on. Defaults to 10.
                num_channels (int, optional): Number of input channels. Defaults to 1.
                beta (int, optional): Weight to control the influence of the KL-divergence on the output of self.loss_function. Defaults to 1."""
        super().__init__(latent_dim, input_height, input_width, num_classes, num_channels)
        self.beta = beta
        
        self.cvencoder = CVEncoder(self.latent_dim, self.input_height, self.input_width, self.num_classes, self.num_channels)
        self.cvdecoder = CVDecoder(self.latent_dim, self.input_height, self.input_width, self.num_classes, self.num_channels)
        
    def forward(self, input, c): 
        mu, sigma = self.cvencoder(input, c)
        z = self.reparametrization(mu, sigma)
        x_hat = self.cvdecoder(z, c)
        return x_hat, mu, sigma
    
    def reparametrization(self, mu, sigma):
        if self.training: 
            r = torch.normal(0, 1, size=sigma.size()).to(device)
            z = mu + sigma * r
        else: z = mu
        return z
    
    def loss_function(self, x, xhat, mu, sigma): 
        """Loss function to train a CVAutoencoder object (self). 
            Args:
                x (torch.FloatTensor): Input image. 
                xhat (torch.FloatTensor): Output image. 
                mu (torch.FloatTensor): Expectation value of input in latent space. 
                sigma (torch.FloatTensor): Standard devation value of input in latent space. 

            Returns:
                torch.FloatTensor: Loss <-- Binary Cross Entropy + KL-Divergence"""

        BCE = nn.functional.binary_cross_entropy(xhat, x, reduction='sum')
        KLD = -self.beta * 0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

        return BCE + KLD