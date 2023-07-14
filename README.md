GAN Project: Fashion MNIST Image Generation
This project involved building a Generative Adversarial Network (GAN) based on the Fashion MNIST dataset. The objective was to create a model that could generate new images similar to the ones in the dataset and explore arithmetic operations on the generated images.

Project Overview
Dataset Selection: Fashion MNIST dataset was chosen due to its diverse fashion-related images, which provided an opportunity to perform arithmetic operations on the generated images, such as combining shoe and sandal or subtracting dress from a shirt.

Implementation: The GAN model was developed using PyTorch. The project included the creation of three main components:

generator.py: Implements the generator network responsible for generating new images.
discriminator.py: Implements the discriminator network responsible for distinguishing between real and generated images.
loader.py: Handles data loading and preprocessing tasks.
Training: The GAN model was trained using a two-player game approach, where the generator and discriminator networks competed against each other to improve their respective performances. The trained model weights were saved in the gans_weights.pth file.

Image Generation and Analysis: To evaluate the model's performance, the generator was fed random 100-dimensional noisy vectors, sampled from a Gaussian distribution. The generator then produced 100 generated images, which were plotted and analyzed. Additionally, arithmetic operations were performed on the noisy vectors to demonstrate the resulting image differences, both in terms of pixel-wise subtraction or addition and vector-wise operations.

Usage
To use this program, follow these steps:

Download the repository.
Run myScript.py to execute the program.
The program will generate new images based on the trained GAN model and perform arithmetic operations on the generated images.
Technologies Used
Python
PyTorch
Please note that the complete results, including screenshots, can be found in the provided PDF document.

Feel free to customize the description, technologies used, and usage instructions based on your specific project details.
