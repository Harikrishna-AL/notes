---
title: "Fuzzy_c"
date: 2023-02-21T22:45:25+05:30
draft: true
---

# Fuzzy C
Fuzzy C is a model which sofly classifies data points into clusters, which means that a single datapoint can belong to multiple clusters according to it's probability of being in those classes.

#  Model data flow
- #### Flowchart

    ![image](https://user-images.githubusercontent.com/91690484/216776804-482c6832-b80b-4141-9d7b-6e7b25319a97.png)
    
- #### Membership value

  Here Mu represents the membership value. Mu(ij) means that membership value of i(th) point in the k(th) cluster. This value is calculated for each data     points in every interation and stored in a matrix. Also, the cluster centers are also decided on the this basis at the end of every iteration.

  ![image](https://user-images.githubusercontent.com/91690484/216777309-fa0acf48-3fe5-47c1-a247-f5508baefb28.png)




