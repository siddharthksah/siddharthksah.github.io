---
title: 'Real time Object tracking and Segmentation using YoloV8 with Strongsort, Ocsort and Bytetrack'
date: 2023-02-09
permalink: /posts/2023/02/real-time-object-tracking-and-segmentation-using-yolo-v8-with-strongsort-ocsort-and-bytetrack/
tags:
  - Object Tracking
  - Computer Vision
  - Object Detection
---
<style>
    .blog-intro {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: Arial, sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #333;
        background-color: #f8f8f8;
        padding: 20px;
    }

    .intro-text {
        flex: 1;
        font-weight: bold;
    }

    .intro-image {
        flex: 1;
        margin-left: 20px;
        text-align: right;
    }

    .intro-image img {
        width: 100%;
        border-radius: 8px;
    }

    .image-caption {
        font-size: 8px;
        color: #666;
        margin-top: 0px;
    }
</style>

<div class="blog-intro">
    <div class="intro-text">
        <p>
            The goal of object tracking is to keep track of an object as it moves through the frame and to locate it in subsequent frames. In this articlewe will compare different types of algorithms and see how to implement one of them.
        </p>
    </div>
    <div class="intro-image">
        <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/posts/real-time-object-tracking-and-segmentation-using-yolo-v8-with-strongsort-ocsort-and-bytetrack_2.gif?raw=true">
        <p class="image-caption"><em>Image generated using text-to-image model by Adobe</em></p>
    </div>
</div>

![](/Users/siddharthsah/Desktop/siddharthksah.github.io/posts/real-time-object-tracking-and-segmentation-using-yolo-v8-with-strongsort-ocsort-and-bytetrack_1.gif)


The goal of object tracking is to keep track of an object as it moves through the frame and to locate it in subsequent frames.

There are various methods for object tracking, including:

*   Feature-based tracking: This method involves tracking an object based on its features, such as colour, shape, or texture.
*   Template matching: This method involves matching a pre-defined template to each frame in the video sequence.
*   Correlation-based tracking: This method involves computing the similarity between the target object and candidate regions in subsequent frames.
*   Deep learning-based tracking: This method uses neural networks trained on large datasets to detect and track objects in real-time.

Object tracking has many potential applications, including:

*   Video surveillance: Object tracking can be used in security systems to track objects of interest, such as vehicles or people, in real-time.
*   Human-computer interaction: Object tracking can be used to detect and track body movements in order to enable gesture recognition.
*   Robotics: Object tracking can be used to enable robots to track and follow objects, such as people or objects of interest.
*   Sports analysis: Object tracking can be used to track athletes and analyse their performance in sports such as soccer or basketball.
*   Automated driving: Object tracking can be used in autonomous vehicles to track other vehicles, pedestrians, and road signs in real-time.

The tracking is divided into two parts. The first one is detecting the object and the second one is using an algorithm to track that object in subsequent frames.

Object detection using YOLO
--------

YOLO (You Only Look Once) is a popular object detection algorithm that uses a convolutional neural network (CNN) to detect objects in images and videos. YOLO was developed to be fast and efficient, and it can detect objects in real-time on a standard computer.

The YOLO algorithm works by dividing an image into a grid of cells, and each cell is responsible for predicting the presence of objects within it. If an object spans multiple cells, each cell responsible for the object predicts the presence of the object. The YOLO network then predicts the bounding box coordinates and class probabilities for each object in the image.

One of the main advantages of YOLO is its speed. YOLO uses a single CNN to make predictions for an entire image, which makes it faster than other object detection algorithms that use multiple CNNs or a sliding window approach. Additionally, YOLO uses anchor boxes to handle the problem of object aspect ratios, which helps it to detect objects with varying shapes.

YOLO has been trained on a large dataset and can detect a wide range of objects, including people, cars, animals, and more. YOLO is commonly used for a variety of applications, including object tracking, video surveillance, and autonomous vehicles.

YOLO is a powerful object detection algorithm that is fast, efficient, and capable of detecting a wide range of objects. If you’re interested in using YOLO for object detection, there are several pre-trained models available that you can use, or you can train your own model on your own dataset.

We will be using yolov8.

YOLOv8 is the latest version of the YOLO (You Only Look Once) object detection algorithm. It builds on the previous versions of YOLO and has several new features and improvements.

Here are some of the new features and improvements in YOLOv8:

1.  Improved architecture: YOLOv8 has a new architecture that is designed to be more efficient and accurate than previous versions. This includes a new backbone network and a more effective use of anchor boxes.
2.  Better object detection: YOLOv8 includes improved object detection capabilities, with a higher accuracy and improved ability to detect small objects.
3.  Increased speed: YOLOv8 is faster than previous versions, making it suitable for real-time applications such as video surveillance and autonomous vehicles.
4.  Enhanced performance on various platforms: YOLOv8 has been optimized for various platforms, including GPUs, CPUs, and mobile devices, making it more versatile and accessible for a wider range of applications.
5.  Improved data augmentation: YOLOv8 includes improved data augmentation techniques to increase the robustness of the model and prevent overfitting.

YOLOv8 is a significant improvement over previous versions of YOLO and represents the state-of-the-art in real-time object detection. If you’re interested in using YOLOv8 for object detection, there are several pre-trained models available, or you can train your own model on your own dataset.

We have 3 different types of trackers namely StrongSORT OSNet, OCSORT and ByteTrack.

StrongSORT is an object tracking algorithm that is based on a combination of deep neural networks and traditional computer vision techniques. StrongSORT was developed to overcome some of the limitations of other object tracking algorithms and to provide improved accuracy and robustness in challenging tracking scenarios.

One of the key features of StrongSORT is its use of deep neural networks to extract features from an image that are robust to appearance changes, such as changes in illumination, viewpoint, and occlusion. StrongSORT also uses a Kalman filter to model the motion of the target object and to estimate its position in subsequent frames.

Another advantage of StrongSORT is its ability to handle occlusions and partial occlusions, which can be a challenging problem for other object tracking algorithms. StrongSORT uses a combination of feature matching and Kalman filtering to handle occlusions and estimate the position of the target object even when it is partially occluded.

StrongSORT is a powerful object tracking algorithm that provides improved accuracy and robustness in challenging tracking scenarios. If you’re interested in using StrongSORT for object tracking, there are several implementations available, or you can implement your own version based on the published research.

OCSORT (Online Continuous-Time Object Tracking) is an online object tracking algorithm that is designed to track objects in real-time. It is designed to handle challenging tracking scenarios such as occlusions and changes in object appearance.

One of the key advantages of OCSORT is its real-time performance. OCSORT is designed to track objects in real-time, even in complex and cluttered scenes, and it is capable of handling high-frame rate videos.

OCSORT is a powerful object tracking algorithm that provides real-time performance and high accuracy in challenging tracking scenarios.

ByteTrack was developed to provide improved accuracy and robustness in object tracking, and it has been applied to a wide range of tracking scenarios, including person tracking, vehicle tracking, and drone tracking.

One of the key advantages of ByteTrack is its use of reinforcement learning to train a deep neural network to perform object tracking. Another advantage of ByteTrack is its ability to track objects in real-time. ByteTrack is designed to run in real-time on a standard computer or even on mobile devices, and it has been shown to achieve high frame rates and low latency in a range of tracking scenarios.

ByteTrack is a powerful object tracking algorithm that provides improved accuracy and robustness in challenging tracking scenarios. If you’re interested in using ByteTrack for object tracking, there are several implementations available, or you can implement your own version based on the published research.

Let’s talk code.

```python
# Create a virtual environment named "tracking" with Python 3.7
conda create -n tracking python=3.7 -y

# Activate the "tracking" virtual environment
conda activate tracking

# Clone the YOLOv8 tracking repository recursively
git clone --recurse-submodules https://github.com/mikel-brostrom/yolov8_tracking.git

# Navigate to the yolov8_tracking directory
cd yolov8_tracking

# Upgrade pip to ensure the latest version
pip install --upgrade pip

# Install the required dependencies from the requirements.txt file
pip install -r requirements.txt

# Perform object detection and tracking using YOLOv8 on the video "output.mov" and save the result
python3 track.py --source output.mov --yolo-weights yolov8s.pt --save-vid

# Perform object detection with segmentation and tracking using YOLOv8 on the video "output.mov" and save the result
python3 track.py --source output.mov --yolo-weights yolov8s-seg.pt --save-vid

# Select the tracking method "StrongSORT" and perform object tracking on the live video stream (source 0)
python3 track.py --tracking-method strongsort --source 0

# Select the tracking method "OCSORT" and perform object tracking on the live video stream (source 0)
python3 track.py --tracking-method ocsort --source 0

# Select the tracking method "ByteTrack" and perform object tracking on the live video stream (source 0)
python3 track.py --tracking-method bytetrack --source 0

```

YoloV8 Re-identification weights

```python
python3 track.py --yolo-weights yolov8s.pt --reid-weights osnet_x0_25_msmt17.pt --source 0 --save-vid
```

ReID (Re-Identification) weight is a term used in the context of object tracking algorithms. It refers to the weight assigned to the ReID (Re-Identification) loss term in the objective function of the tracking algorithm.

ReID loss is used in object tracking algorithms to ensure that the target object is accurately re-identified in subsequent frames of a video. The ReID loss term compares the appearance of the target object in the current frame to the appearance of the target object in previous frames. The ReID weight determines the importance of this term in the overall objective function of the tracking algorithm.

In some object tracking algorithms, the ReID weight is learned during training, while in others it is set based on prior knowledge or heuristics. The ReID weight can be critical to the performance of the object tracking algorithm, as it determines the balance between the importance of accurately re-identifying the target object and the importance of accurately estimating its motion.

In general, setting the ReID weight too low may result in poor re-identification performance, while setting it too high may result in poor motion estimation performance. The optimal ReID weight will depend on the specific scenario and requirements of the tracking problem.

<div style="max-width: 100%;">
  <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/posts/real-time-object-tracking-and-segmentation-using-yolo-v8-with-strongsort-ocsort-and-bytetrack_1.gif?raw=true" style="width: 100%; height: auto;">
</div>


