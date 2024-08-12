---
title: "Taking Inspiration from Brain 🧠"
date: 2024-06-02T17:38:03+05:30
draft: true
---
This note in short explains the different processes learning may take place in the brain. What I want is to inspire some of these phenomenons in the field of AI and create some interesting neural networks.

## why do we need to understand this?
I personally believe that the most of the world and major startups are stuck in the same loop of trainig LLMs while it is quite evident that it’s a dead end. We might never reach the complexity of human brain using the transformers architecture.

## Some things we can learn from biology
- Brain makes new connections based on experiences
- The synapses gets stronger when some sensory input triggers it frequently
- Memories are only stored in certain areas of brain and all neurons are not capable of storing memory
- A single neuron can interact with other neurons which are really far away and thus could move away from the traditional approach of giving the neighboring nodes more importance. 
- There are multiple connections between the same two neurons and the strength of the connection is decided by the frequency of it’s activation.

## Exploring them one by one
- **Making newer connections**: This topic is already explored by few papers and is known as the concept of “neurogenesis”. I myself tired creating a framework of growing Neural Network using attention based mechanism and I will be sharing the results soon! The idea here is that certain rules can be defined which neurons can either be added or connections can be pruned. Also, we need to design the framework in such a way that with just fewer training loops, we can re-train the network to adapt to the new connections and new knowledge.

- **Synaptic Strengthening**: This is the most interesting part of the brain. The synapses are not just connections but they are the ones which store the memory. The more frequently a synapse is triggered, the stronger it becomes. This is the concept of “Hebbian Learning”. The idea here is to create a mechanism where the weights of the synapses are updated based on the frequency of the activation of the neurons. This can be done by creating a mechanism where the weights are updated based on the frequency of the activation of the neurons. Maybe we could give an importance factor to neurons based on the frequency of their activation and then update the weights of the synapses based on the importance factor.

- **Memory Storage**: The memories are mostly stored in certain areas of the brain. This is the concept of “Sparse Memory”. The idea here is to create a mechanism where only certain neurons are capable of storing memory and the rest of the neurons are just used for processing. This can be done by creating a mechanism where the neurons are divided into two groups, one for storing memory and the other for processing.

- **Long Range Connections**: The brain is capable of making long-range connections between neurons. In the implementations I saw of growing networks, the connections were updated through GraphConv layers which only took the neighboring nodes into consideration. Therefore, I tried using **Edge-Augmented Attention** to update the node embedding and edges based on the values of all the nodes in the graph. This way, the network can make long-range connections between neurons. Unfortunately, I couldn’t get the results as growing the graph beyond fewer neurons threw error. This was because I had used PCA to reduce dimensionality of the edge matrix but it gave error for larger matrices. Here is an example of the growing network I tried to create by taking inspiration from the NDP paper.


    ![Image alt](array2.gif)


