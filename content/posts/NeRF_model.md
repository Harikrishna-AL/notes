---
title: "NeRF"
date: 2023-04-23T21:57:32+05:30
toc: true
draft: true
---




- ## _**what are NeRFs?**_
    NeRFs are a type of network used to generate unseen views of an object using the seen views as the dataset. The model takes a 5D(3d coordinates,camera angles) input and then outputs a 4D(rgb color, densoty) vector. 
    <!-- <div style="align:center"><img src="./images/nerf.jpeg" alt="no image"></div> -->
- ## _**Sampling Inputs**_
    The 3 dimensinal coordinates that has to be passed into the neural network is sampled using a ray casting algorithm. The algorithm is as follows:   
    1. Cast a ray from a pixel of the image to the object.
    2. Sample points along the ray.
    3. Pass the sampled points to the network.

    ![Image alt](ray_cast.png)
    
    Code:
    ```
    def compute_rays(height,width,focal_length,cam2world):
        x ,y = tor_to_meshgrid(
            torch.arrange(width).to(cam2world),
            torch.arrange(height).to(cam2world)
        )
        directions = torch.stack(
            [(x - width*0.5)/focal_length,
            (y - height*0.5)/focal_length,
            torch.ones_like(x)]
        )
        origins = torch.broadcast_to(
            cam2world[:3,-1],
            directions.shape
        )
        rays_d = torch.sum(
            directions[...,None,:] * cam2world[:3,:3],
            dim=-1
        )
        rays_o = torch.sum(
            origins[...,None,:] * cam2world[:3,:3],
            dim=-1
        )
        return rays_o,rays_d
    ```
    
- ## _**Archeitecture**_
    <!-- making image centered -->
    <!-- <div style="align:center"><img src="./nerf.jpeg" alt="no image"></div> -->
    ![Image alt](nerf.jpeg)
    <!-- {{< figure src="nerf.jpeg" title="NeRF" caption="asdfghjk" class="center" >}} -->

    The network is a fully connected network with 8 layers. The input to the network is a 5D vector and the output is a 4D vector. The network is trained using a loss function which is the sum of the MSE of the rgb color and the density. The network is trained using the Adam optimizer. 
