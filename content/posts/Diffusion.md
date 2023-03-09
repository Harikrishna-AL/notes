---
title: "Diffusion"
date: 2023-02-21T22:39:12+05:30
draft: true
---

# **Denoising Diffusion Probablistic Models**

This note is about the diffusion model. It will cover about it's working and implementation both mathematically and code wise. Diffusion is one of the latest techniques used to generate new images by basically destroy them by adding more and more noise. So a basic diffusion model will contain a noise scheduler, a neural network to predict the noise in an image at given timestamp (t) (U-Net is this case) and a loss function which enables the model to learn from dataset.

- ## Noise Scheduler (Forward Diffusion)
    The job of the noise scheduler is to add noise into the image during different timestamps by increasing it in a linear, quadratic and other fashions. We'll be using linear in our case.
    This adds noise to the image keeping mean as _**sqrt(1 - beeta)**_ and variance as _**beeta**_. This can be seen similar to a markov chain as the noise in the image at _t_  depends on the pixels of the image on time _t - 1_.
    
    <div align=center><img src="https://user-images.githubusercontent.com/91690484/224030024-eac15771-c98b-4e56-af71-e32b0c9f3138.png"></div>
    
    This diagram shows the diffusion process. It's a chain process that adds noise step by step till the image is completely destroyed and is dependent on the parameter <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>&#x3B2;</mi></math> and α = 1 - <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>&#x3B2;</mi></math>
    
    <div align=center><img src="https://user-images.githubusercontent.com/91690484/224036368-b3214071-9868-40a4-8cf8-35d11f2f1a86.png"></div>
    
    This formula gives the image at time t given the image at time t=0. The parameters here are the cumilative sum of the parameters till time t.


- ## Reverse Diffusion 
    Now we need a neural network to predict the noise in a given image at particular timestamp t. We'll be using U-Net because it has the ability to use the information in the previous adjacent latent space. This gives the network attention and makes it possible to predict noise. The model is also given the timestamp as input by converitng it into a time embedding and adding it to different stages of the model.
- ## Training



