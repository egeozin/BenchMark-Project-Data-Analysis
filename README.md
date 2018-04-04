

# BenchMark

http://benchmark.mit.edu/

One of the most challenging problems facing urban designers and planners today is the difficulty of developing robust and effective methods of studying public spaces that result in more informed decision making. At its core, this problem revolves around measuring *Public Life*, a term often referenced in Jan Gehl’s work that refers to citizens’ daily interactions with others within the built environment.

The Benchmark project investigates how we can use technology to better measure and analyze the qualitative elements of public life. More specifically, by tracing human interaction in space, we can begin to understand quantitatively what makes a “good” urban space and then develop metrics that can be used to improve the public realm. Benchmark begins to address this question by introducing mobile urban furniture embedded with sensors that simultaneously create flexible public spaces and collect information about how these spaces operate socially.

<p align="center"><img src="https://github.com/egeozin/BenchMark-Project-Data-Analysis/blob/master/imgs/observation.png" width="500"/></p>

The Gehl method entails a researcher observing individuals’ movements through space and tracking their behavior. A researcher looks at a person and tracks his or her position and activities over time. This manual process is generally superior to digital analysis because state-of-the-art computer vision algorithms have limited ability to determine whether one individual in a picture shows up in a series of sequential frames.

For this challenge we set out to attempt to develop a system that tracks the behavior of individuals through time without identifying them. Furthermore, we aimed to capture temporal patterns of inhabitation such as individuals' use of, and interaction within the space and with other individuals in the space. As a result, we used the GoPro cameras embedded in one of our furnitures to collect images of the place of interest.

## Methods

Our primary goal was to locate people moving through the camera’s field of view on the 2D ground plane. This helped us track how people move through the space and identify areas of particular interest. Initially, we needed to differentiate people from objects. To address that, we tested two facial recognition algorithms, the Haar Cascade Classifiers and the Support Vector Machines, that both yielded unsatisfactory mAP values in our test set. Instead, we decided to use a type of Convolutional Neural Network (CNNs) algorithm.

We decided to use the Faster R-CNN (Regions with CNN) algorithm with a pre-trained model that is trained with a large set of images belonging to certain classes. The main difference between a CNN and the Faster R-CNN method is that the latter is calibrated to detect multiple objects and their boundaries in an image.

## Analysis

Utilizing Faster R-CNN for detecting people yielded vastly superior results. We used the results derived from the Faster R-CNN algorithm to compute the number of people present at our site in a single time frame and whether they stayed on the site over multiple time frames that in total comprised a duration less 30 seconds.

<p align="center"><img src="https://github.com/egeozin/BenchMark-Project-Data-Analysis/blob/master/imgs/gopro_combined.gif" width="500"/></p>

Our benches have a unique design, as a result, the Faster R-CNN algorithm was not able to detect them as benches. In order to achieve this goal, initially, we individually tagged the benches in the GoPro images. More specifically, we sorted through the GoPro images and noted every time a bench moved, drawing a bounding box around that bench’s new location. With that data in hand, we wrote a script that calculates the relative distance of the bounding box to the camera and returns the coordinates of the bench. Using this script, it was possible to compute the correct location of the benches and the amount of time each bench stayed in a given location.

However, as this process turned out to be time-consuming, we came up with a better method. We created a dataset from a set of images of our benches taken from different angles and scales, as well as in different light conditions. Additionally, we compiled a training image dataset where we labeled people in bounding boxes based on how they are using the benches (e.g., whether as seats or backrests).  We used these datasets to train a new neural network, named “You Only Look Once” (Girshick et al., 2016). We chose this algorithm because it enabled us to train it with our custom labeled data faster and it’s as robust as the previous algorithm (Faster R-CNN).  As a result, this network provided us with proper labeling of the benches and the people who used them as backrest or seating along with their corresponding bounding boxes.

<p align="center"><img src="https://github.com/egeozin/BenchMark-Project-Data-Analysis/blob/master/imgs/detector_compare.png" width="700"/></p>

The table above shows the comparison of the visual detectors utilized in the Benchmark project. For the Benchmark project we trained a pre-trained YOLO network with a novel dataset. Our new dataset contained 150 images belonging to each new classes (bench, sit, backrest). Usually, in order to train neural networks with new classes, compiling a dataset with more than 1000 examples for each class is considered good practice. However, the detection performance we got from the YOLO network was promising given that YOLO method helps us to considerably increase the mean average precision (mAP) with smaller amount of data compared to other methods. Furthermore, in its current state, the YOLO method which we trained with our dataset proven to be suitable to be deployed to the various mobile devices (e.g. Raspberry Pi). The reason is that we can trade-off a certain amount of accuracy against the speed-up we gain in processing frames which is shown to be equally important in this setting. Moreover, in terms of the number of frames processed per second (FPS), YOLO is superior compared to the Faster R-CNN algorithm. This allows us to reliably compute necessary predictions in the field without storing the images in a temporary database. Finally, we used these predictions for calculating the number of pedestrians visiting the site and computing various indices for measuring public interaction. We used these indices to measure stationary interactions.

### Stationary Interactions

Stationary interactions are key proxies for measuring the performance of public space. By asking questions on whether people are sitting on the benches and how people are moving the benches around allowed us to better understand how people are using the space.

Our results show that, people tend to cluster their public seating. While researchers have observed this clustering effect in the context of fixed street furniture, we sought to understand what level of clustering might occur when the constraint of fixed seating is lifted. We determined that people will take advantage of mobile street furniture and situate it in a way that actively promotes social interactions. Thus, we defined bench clusters as two or more benches close enough to facilitate social interaction, which we estimated to be 1.5m, and then identified locations in the public spaces where bench clustering occurred most frequently, and length of time when benches remained clustered with one another. Furthermore, by combining the cluster calculations with the pressure data, we derived a unique insight that begins to point to the origins and evolution of social interaction in public space. By adding information about the amount of time people were seated on each bench, we can determine how the configuration of benches impacts bench usage. What we found was that not only are benches more likely to be clustered throughout the day, people are more likely to sit on them when they are clustered. This has valuable implications for the built environment, as urban designers who desire a highly social public plaza may want to consider clustering concentrated nodes of seating in the space rather than dispersing benches across it.

<p align="center"><img src="https://github.com/egeozin/BenchMark-Project-Data-Analysis/blob/master/imgs/bench_combined.gif" width="500"/></p>





