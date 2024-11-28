Anime Face Generation using DCGANs
-------------------------------------------------------------
The primary challenge addressed in this project is the generation of high-quality, 
realistic anime faces from random noise


Overview
-------------------------------------------------------------
This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for generating anime face images. The project utilizes TensorFlow and PyTorch for model training and evaluation, as well as TensorFlow Hub for pre-trained models.

Features
-------------------------------------------------------------
- Dataset Handling: Automatically downloads the Anime Face dataset from Kaggle.
- Image Preprocessing: Reshapes and normalizes images for GAN training.
- GAN Model: Includes the implementation of both the Generator and Discriminator   models.
- Training Loop: Custom training loop for GANs with metrics and callbacks.
- Evaluation: Calculates Fréchet Inception Distance (FID) score using PyTorch.

Setup and Installation
-------------------------------------------------------------
1. Clone the Repository using:
    ``` 
        git clone https://github.com/AbdulHaq1503/GAN-Project
        cd GAN-Project

    ```
2. Install the required dependencies:  
    Note: Ensure you have pip and a Python environment (e.g., virtualenv or Conda) set up before running this command.
    ```
        pip install tensorflow tensorflow-hub
        pip install torch torchvision pytorch-fid

    ```
3. Download Kaggle Dataset
    Ensure that you have a Kaggle account and API key configured. The script uses kagglehub to download the dataset. Install the library if needed:
    ```
    pip install kagglehub
    import kagglehub
    dataset_path = kagglehub.dataset_download('soumikrakshit/anime-faces')

    ```
Dataset
-------------------------------------------------------------
 Contains 21,551 Anime Face images scraped from www.getchu.com
 Images shaped 64 x 64 x 3
 Images contain human and non-human faces
 Link to dataset : https://www.kaggle.com/datasets/soumikrakshit/anime-faces/code

How It Works
-------------------------------------------------------------
1. Input Latent Vector:
A random noise vector of size LATENT_DIM=100 is sampled as input to the Generator.

2. Generator:
Transforms the noise vector into an image (64x64x3) through a series of Conv2DTranspose layers.
Outputs an image with pixel values normalized between -1 and 1 (using Tanh activation).

3. Discriminator:
Takes an input image (real or fake) and evaluates its authenticity.
Processes the image through Conv2D layers and outputs a probability score (real or fake).

4. Training:
The Generator tries to fool the Discriminator by generating realistic images.
The Discriminator learns to distinguish real images from fake ones.
Both models are trained simultaneously in an adversarial setup.

5. Loss Functions:
Generator: Maximizes the probability of the Discriminator classifying generated images as real.
Discriminator: Minimizes the difference in predictions for real and fake images.
Evaluation:

6. The Fréchet Inception Distance (FID) is used to measure the similarity between generated images and real images.

Hyperparameters
-------------------------------------------------------------
The key hyperparameters used in the project are:

1. LATENT_DIM :	100	Size of the random noise vector input to the generator.
2. Generator Learning Rate : 0.0003
3. Generator Learning Rate : 0.0001
4. Number of epochs for training the DCGAN : 50	

Architecture Description
-------------------------------------------------------------
Generator : 
1. Dense Layer + Reshaping : Filter Shape : 8 x 8 x 512
2. Conv2DTranspose Layer: Filter Shape : (4, 4, 256)
3. Conv2DTranspose Layer: Filter Shape : (4, 4, 128)
4. Conv2DTranspose Layer: Filter Shape : (4, 4, 64)
All above filters use ReLU activation function with (2,2) stride
5. Conv2D Layer: Filter Shape : (4, 4, 3) with Tanh activation

Discriminator :
1. Conv2D Layer: Filter Shape : (4, 4, 64)
2. Conv2D Layer: Filter Shape : (4, 4, 128)
3. Conv2D Layer: Filter Shape : (4, 4, 128)
All above filters use LeakyReLU (alpha = 0.2) activation function with (2,2) stride
4. Dense Layer with Sigmoid activation function

Outputs
-------------------------------------------------------------
Synthetically Generated Anime Images: Saved in the generated_images/ directory.
Real Images: Saved in the real_images/ directory for FID comparison.
Saved Generator: The trained generator model is saved as generator.h5.
FID Scores: Calculated at the end of each epoch and printed to the console.

Results
-------------------------------------------------------------
The model attains:
1. FID score : 35.97 (70 epochs)
2. Generator Loss : 1.4053
3. Discriminator Loss : 0.5203

