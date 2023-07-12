import torch
import matplotlib.pyplot as plt
from Generator import Generator
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 100  # z vector
num_samples_per_class = 10  # Number of images to generate per class

# Load the trained model
img_shape = 28 * 28  # for FashionMNIST

def generate_and_plot_images(generator, num_images_per_class):
    num_classes = 10
    #plot 100 generated images
    fig, axs = plt.subplots(num_classes, num_images_per_class, figsize=(10, 10))

    for i in range(num_classes):
        for j in range(num_images_per_class):
            class_label = i
            z = torch.randn(1, latent_dim).to(device)  # random vector from a Gaussian distribution size like z
            z[:, class_label] = 1.0  # Set the class label for the desired class to 1
            generated_image = generate_class_x_image(z, generator,latent_dim, device)
            axs[i, j].imshow(generated_image, cmap='gray')  # show in gray scale
            axs[i, j].axis('off')

    plt.show()

def generate_class_x_image(z,generator, latent_dim,device):
    # Generate fake image from the given latent vector
    with torch.no_grad():
        generated_image = generator(z).cpu()

    # Denormalize the generated image to be in the range [0, 1] back from [-1,1] tanh function
    generated_image = (generated_image + 1) / 2

    return generated_image.squeeze()


def plot_difference(symb,class2,class1,c1_image, c2_image, sub_img):
    difference_image = c2_image - c1_image # diffrence pixel wise

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    #img class1
    axs[0].imshow(c1_image, cmap='gray')
    axs[0].axis('off')
    axs[0].set_title("class1")
    #img class2
    axs[1].imshow(c2_image, cmap='gray')
    axs[1].axis('off')
    axs[1].set_title("class2")
    #img 2 - img 1
    axs[2].imshow(difference_image, cmap='gray')
    axs[2].axis('off')
    axs[2].set_title('class2 {} class1 pixel wize'.format(symb))
    #z1- z2
    axs[3].imshow(sub_img, cmap='gray')
    axs[3].axis('off')
    axs[3].set_title('class2 {} class1 vector z'.format(symb))

    plt.show()

def average_and_subtract(class1, class2,generator, latent_dim, device):

    z0 = torch.randn(1, latent_dim).to(device)
    z0[:, class1] = 1.0


    z1 = torch.randn(1, latent_dim).to(device)
    z1[:, class2] = 1.0

    #z2 = torch.randn(1, latent_dim).to(device)
    z2 = z1 - z0
    #generate images
    c1_image = generate_class_x_image(z0,generator,latent_dim,device)
    c2_image = generate_class_x_image(z1,generator,latent_dim,device)
    sub_img = generate_class_x_image(z2,generator,latent_dim,device)
    #plot the diffrence
    plot_difference("-",class2,class1,c1_image, c2_image, sub_img)

def average_and_add(class1, class2,generator, latent_dim, device):
    #class_labels = [class1, class2]  # Sneaker and Ankle Boot class labels
    z0 = torch.randn(1, latent_dim).to(device)
    z0[:, class1] = 1.0

    z1 = torch.randn(1, latent_dim).to(device)
    z1[:, class2] = 1.0

    z2 = z1 + z0

    c1_image = generate_class_x_image(z0,generator,latent_dim,device)
    c2_image = generate_class_x_image(z1,generator,latent_dim,device)
    sub_img = generate_class_x_image(z2,generator,latent_dim,device)
    #plot the diffrence +
    plot_difference("+",class1,class2,c1_image, c2_image, sub_img)


if __name__ == "__main__":
    generator = Generator(latent_dim, img_shape).to(device)
    generator.load_state_dict(torch.load('gans_weights.pth'))
    generator.eval()
    #ploting 100 generated images
    generate_and_plot_images(generator, num_samples_per_class)

    generator1 = Generator(latent_dim, img_shape).to(device)
    generator1.load_state_dict(torch.load('gans_weights.pth'))
    generator1.eval()

    for k in range (5):
        #two random classes
        class1 = random.randint(0, 9)
        class2 = random.randint(0, 9)

        # subtract them pixel wise AFTER generating and before with vector z
        average_and_subtract(class1, class2, generator1, latent_dim, device)
    for k in range(5):
        #two random classes
        class1 = random.randint(0, 9)
        class2 = random.randint(0, 9)
        #add them pixel wise AFTER generating and before with vector z
        average_and_add(class1,class2,generator1,latent_dim, device)