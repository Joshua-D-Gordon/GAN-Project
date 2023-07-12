import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from Generator import Generator
from Discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0002
batch_size = 64
latent_dim = 100
img_shape = 28 * 28
epochs = 100

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize data between -1 and 1
])

train_dataset = FashionMNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


if __name__ == "__main__":

    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    adversarial_loss = nn.BCELoss()

    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # Configure input
            real_imgs = imgs.view(batch_size, -1).to(device)

            # Train Generator
            generator_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_dim).to(device)
            gen_imgs = generator(z)

            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            generator_optimizer.step()

            # Train Discriminator
            discriminator_optimizer.zero_grad()

            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            discriminator_optimizer.step()

        print(f"Epoch [{epoch}/{epochs}] Loss_G: {g_loss.item():.4f} Loss_D: {d_loss.item():.4f}")

    torch.save(generator.state_dict(), "gans_weights.pth")
    torch.save(discriminator.state_dict(), "gans_weights_discriminator.pth")