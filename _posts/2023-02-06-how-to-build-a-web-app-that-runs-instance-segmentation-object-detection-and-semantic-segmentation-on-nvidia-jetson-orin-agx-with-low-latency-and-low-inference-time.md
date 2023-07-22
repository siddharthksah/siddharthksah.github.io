---
title: 'How to build a web app that runs instance segmentation, object detection and semantic segmentation on Nvidia Jetson Orin AGX with low latency and low inference time'
date: 2023-02-06
permalink: /posts/2023/02/how-to-build-a-web-app-that-runs-instance-segmentation-object-detection-and-semantic-segmentation-on-nvidia-jetson-orin-agx-with-low-latency-and-low-inference-time/
tags:
  - Web App
  - Nvidia Jetson Orin
  - Computer Vision
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
            In article, we will explore to build and host a webapp on Nvidia Jetson Orin edge device for object detection and segmentation task. We will focus on making it low latency and speeding up the inference time.
        </p>
    </div>
    <div class="intro-image">
        <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/posts/navigating-the-maze-streamlined-project-structure-for-data-science-and-everything-around-it.jpeg?raw=true">
        <p class="image-caption"><em>Image generated using text-to-image model by Adobe</em></p>
    </div>
</div>

One of the fastest ways to build a web app that works on almost all platforms is by using the Streamlit library in Python.

Streamlit is an open-source framework for building interactive data science and machine learning applications in Python. It allows developers to create beautiful and fully interactive web applications with just a few lines of Python code.

Uses:

*   Creating data exploration tools and dashboards
*   Building machine learning models and prototypes
*   Creating visualizations, animations and other interactive content
*   Sharing and collaborating on data science projects with others

Streamlit uses a simple Python API to build and run web applications, which are served by a built-in web server. The framework provides a variety of built-in widgets for user inputs, like sliders, dropdowns, and text inputs, as well as charting and visualization libraries to display data. The application code is written in Python and Streamlit automatically handles the underlying front-end and back-end technologies, like HTML, JavaScript, and CSS, so developers can focus on building their application logic.

When a user interacts with the application, Streamlit updates the front-end and re-runs the necessary Python code to update the displayed content, making it a very powerful and flexible tool for data science and machine learning applications.

TL;DR Streamlit rocks!

What’s Nvidia Jetson Orin AGX? Why did I use it?

Nvidia Jetson Orin AGX is a high-performance, low-power AI computer platform designed for use in robotics and autonomous machines. It is equipped with powerful GPUs, specialized hardware engines, and software tools optimized for AI applications, making it ideal for building autonomous systems with demanding real-time requirements. The platform provis a comprehensive solution for AI applications including computer vision, deep learning, and robotics, reducing the time and effort required to bring new AI-powered products to market.

Here are some key advantages of Nvidia Jetson Orin AGX:

1.  High Performance: Jetson Orin AGX features powerful GPUs and specialized hardware engines, allowing it to perform complex AI tasks at high speeds and with low power consumption.
2.  Low Power Consumption: The platform is designed to be energy-efficient, making it ideal for use in battery-powered devices and systems that require long periods of operation.
3.  Easy Integration: Jetson Orin AGX integrates easily with other hardware and software components, reducing the time and effort required to bring new AI-powered products to market.
4.  Scalability: Jetson Orin AGX can be easily scaled to meet the needs of a wide range of applications, from small, single-board systems to large, multi-board configurations.
5.  Compact Form Factor: The platform has a small form factor, making it easy to integrate into a wide range of devices and systems, including drones, robots, and industrial equipment.
6.  Advanced Features: Jetson Orin AGX includes features such as hardware-accelerated video encoding and decoding, as well as support for multiple cameras and sensors, making it ideal for use in advanced video and imaging applications.

Below is the project structure-

```
 root/---|   
         |---object\_detection/  
         |         |---- weights/  
         |         |---- object\_detection\_demo.py  
         |         |---- assets/  
         |  
         |---instance\_segmentation/  
         |         |---- weights/  
         |         |---- instance\_segmentation\_demo.py  
         |         |---- assets/  
         |  
         |---semantic\_segmentation/  
         |         |---- weights/  
         |         |---- semantic\_segmentation\_demo.py  
         |         |---- assets/  
         |---requirements.txt  
         |---demo\_v12.py  
         |---Dockerfile  
         |---LICENSE.txt  
         |---environment.yml  
         |---README.md
```

[https://github.com/siddharthksah/SUTD-WebApp-Demo/tree/v12](https://github.com/siddharthksah/SUTD-WebApp-Demo/tree/v12)

Here I have used compact DNNs and EfficientNet-based models.

How to install Docker on Nvidia Jetson Orin AGX

To install Docker on Nvidia Jetson Orin AGX, follow these steps:

1.  Update the software sources:

```
sudo apt update
```

2\. Install required packages:

```
sudo apt install -y curl
```

3\. Add the Docker GPG key:

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

4\. Add the Docker repository:

```
sudo add-apt-repository "deb \[arch=arm64\] https://download.docker.com/linux/ubuntu $(lsb\_release -cs) stable"
```

5\. Update the software sources again:

```
sudo apt update
```

6\. Install Docker:

```
sudo apt install -y docker-ce
```

7\. Start the Docker service:

```
sudo systemctl start docker
```

8\. Let’s build Docker container with pytorch for Jetson

```
sudo -s  
docker build -t jetson/pytorch:1.0
```

9\. Running the Docker container

```
nvidia-docker run -it -v {path to the cloned directory}:/home/code/ --add-host=host.docker.internal:host-gateway jetson/pytorch:1.0   
cd home/code
```

How to make the app faster for inferences?

Well there are multiple ways to do it, from using cache to using a different framework altogether. Let’s look at some.

1.  Minimize the number of reruns: Streamlit reruns the entire app every time you change a widget value, which can slow down your app. Minimizing the number of reruns can significantly improve the performance of your app.
2.  Use caching: Streamlit has a built-in caching mechanism that can cache the results of expensive computations. This can significantly reduce the time it takes to rerun your app.
3.  Optimize data processing: Make sure your data processing is optimized and efficient. Consider using Pandas or Dask to process your data efficiently.
4.  Use a smaller data set: If you have a large data set, consider using a smaller sample or subsample of the data during development to speed up the iteration process.
5.  Avoid using global variables: Global variables can cause unexpected behavior and slow down your app.
6.  Use fewer widgets: Try to limit the number of widgets in your app, as each widget can add significant overhead.
7.  Use vectorized operations: Use vectorized operations instead of for loops where possible, as vectorized operations are much faster than for loops.
8.  Consider using a different frontend framework: Streamlit is built on top of the Flask web framework, which can be slow for large and complex applications. If you need more performance, consider using a different frontend framework, such as React or Vue.js.

Practically, one of the ways to get the highest results is to load all the model weights as the app loads. This will lead to a slower app load time but faster inference time. Loading model weights can be significant specially if you have heavier weights >500MB.

Common tricks to reduce inference time in Deep Learning?

1.  Reduce model size: A smaller model requires less memory and computation, leading to faster inference times. Techniques such as pruning, quantization, and knowledge distillation can be used to reduce the size of your model.
2.  Use GPU or TPU: GPUs and TPUs are specialized hardware designed for deep learning and can significantly speed up the inference process.
3.  Optimize network architecture: Choosing a more efficient network architecture, such as MobileNet or ShuffleNet, can reduce inference time.
4.  Batch processing: Batching multiple inference requests together can improve inference time by reducing the overhead of starting and stopping the model for each request.
5.  Avoid redundant computations: Ensure that the model is not performing any redundant computations, such as multiple convolutions on the same feature map.
6.  Use a runtime optimization library: Use a library such as TensorRT or OpenVINO to optimize your model for inference. These libraries can perform optimizations such as layer fusion, kernel auto-tuning, and precision calibration to improve inference time.
7.  Use a pre-trained model: Use a pre-trained model as a starting point and fine-tune it on your data rather than training a model from scratch. This can reduce the amount of computation required during inference.
8.  Parallel processing: Use parallel processing techniques such as data parallelism or model parallelism to distribute the inference workload across multiple processors.

What is multi threading and how to do it in Python?

Multithreading in Python is a technique of running multiple threads (smaller units of a program) concurrently within a single process. Threads can run parallel tasks and make your program run faster.

Here is an example that demonstrates how to use multithreading in Python:

```python
import threading
import time

def first_function():
    """First function to be executed"""
    for i in range(5):
        print("Running first function")
        time.sleep(1)

def second_function():
    """Second function to be executed"""
    for i in range(5):
        print("Running second function")
        time.sleep(1)

# Creating two threads
thread1 = threading.Thread(target=first_function)
thread2 = threading.Thread(target=second_function)

# Start both threads
thread1.start()
thread2.start()

# Wait for both threads to complete
thread1.join()
thread2.join()

print("Both functions have completed")
```

In the above example, two functions `first_function` and `second_function` are created. Then, two threads are created using `threading.Thread` and each thread is assigned a function to be executed. Finally, both threads are started using the `start()` method and then the main thread waits for both threads to complete using the `join()` method.

Let’s build the webapp now.

```python
# after you have cloned the repo create the conda env and install the requirements  
conda create -n demo python==3.8 -y  
conda activate demo  
pip install -r requirements.txt
```

For starters, let’s split the webapp into two different columns. The left side will be for input from the users and the right side we will show the plots and other information.

```python
# necessary libraries

import sys
import time
import random
import glob
import mlconfig
import json
import timeit
import pytz
import os
import shutil
import torch
from PIL import Image
import argparse


from utils.helper import count_net_flops_and_params, load_pruned_model, hybrid_nas_infer
from utils.helper import Customize
from compact_dnn.data.data_loader import build_data_loader
from compact_dnn.utils.config import setup
from utils.helper import delete_temp_directory, topk2output, tensor2image, evaluate_efficient
from efficientnet.source.efficientnet.models.efficientnet import params
from efficientnet.source.efficientnet.metrics import Accuracy, Average
from efficientnet.source.efficientnet import models
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from random import randrange
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime, timezone
import plotly.express as px
import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings("ignore")


##################################################
##################################################
# EfficientNetB0 #################################
##################################################
##################################################


def inference_efficientnet_b0(batch_size, device):

	config = '/home/dso_code/efficientnet/source/configs/imagenet_100_b0.yaml'

	config = mlconfig.load(config)

	print("-------------------------")
	print(config.model.name)
	print("Batchsize: ", batch_size)

	model = config.model()

	weight = "/home/dso_code/efficientnet/weights/imagenet100_b0/best.pth"

	if weight is not None:
		checkpoint = torch.load(weight)['model']
		pretrained_dict = {key.replace(
			"module.", ""): value for key, value in checkpoint.items()}
		model.load_state_dict(pretrained_dict)

	model.to(device)

	image_size = 224  # for effb0

	print("Image Size: ", image_size)

	mflops, mparam = count_net_flops_and_params(
		model, data_shape=(1, 3, image_size, image_size))

	rootlabel = "/home/dso_code/efficientnet/dataset/imagenet100/"
	root = '/home/dso_code/efficientnet/dataset/custom/val'

	valid_loader = Customize(root, image_size, False,
							 batch_size, num_workers=8)
	# for x, y in valid_loader:
	#     print("#############", x.shape)
	dataset_im100 = datasets.ImageFolder(os.path.join(rootlabel, 'val'))
	class_names = dataset_im100.class_to_idx
	class_names = {y: x for x, y in class_names.items()}

	# JSON file

	f = open(os.path.join(rootlabel, 'Labels.json'), "r")
	concrete_classnames = json.loads(f.read())

	all_result = evaluate_efficient(
		model, class_names, concrete_classnames, valid_loader, device)
	# print(all_result[0]['rep']['FPS'])
	# print(all_result[0]['rep']['name'])
	# print(all_result[0]['rep']['prob'])
	# print(all_result[0]['rep']['image'])

	print("FPS: ", all_result)
	print(f"MFLOPS = {mflops}, MPARAM = {mparam}")

	return all_result, (mflops), (mparam)

	'''

	all_result is a list including n batch_result

	To retrieve the result for each batch:
	- FPS: all_result[batch_index]['rep']['FPS']
		type: float  
	Top-K class for one representive image:all_result[batch_index]['rep']['name']
		type: list including k elements 
	Top-K probability for one representive image: all_result[batch_index]['rep']['prob']
		type: list including k elements 
	Image: all_result[batch_index]['rep']['image']
		type: numpy array <3xHxW>

		'''

##################################################
##################################################


##################################################
##################################################
# EfficientNetB3 #################################
##################################################
##################################################

def inference_efficientnet_b3(batch_size, device):
	config = '/home/dso_code/efficientnet/source/configs/imagenet_100_b3.yaml'

	print("-------------------------")
	config = mlconfig.load(config)

	model = config.model()

	print(config.model.name)
	print("Batchsize: ", batch_size)

	weight = "/home/dso_code/efficientnet/weights/imagenet100_b3/best.pth"

	if weight is not None:
		checkpoint = torch.load(weight)['model']
		pretrained_dict = {key.replace(
			"module.", ""): value for key, value in checkpoint.items()}
		model.load_state_dict(pretrained_dict)

	model.to(device)

	image_size = 300  # for effb3
	print("Image Size: ", image_size)

	mflops, mparam = count_net_flops_and_params(
		model, data_shape=(1, 3, image_size, image_size))

	rootlabel = "/home/dso_code/efficientnet/dataset/imagenet100/"
	root = '/home/dso_code/efficientnet/dataset/custom/val'

	valid_loader = Customize(root, image_size, False,
							 batch_size, num_workers=8)
	# for x, y in valid_loader:
	#     print("#############", x.shape)
	dataset_im100 = datasets.ImageFolder(os.path.join(rootlabel, 'val'))
	class_names = dataset_im100.class_to_idx
	class_names = {y: x for x, y in class_names.items()}

	# JSON file

	f = open(os.path.join(rootlabel, 'Labels.json'), "r")
	concrete_classnames = json.loads(f.read())

	all_result = evaluate_efficient(
		model, class_names, concrete_classnames, valid_loader, device)
	# print(all_result[0]['rep']['FPS'])
	# print(all_result[0]['rep']['name'])
	# print(all_result[0]['rep']['prob'])
	# print(all_result[0]['rep']['image'])
	print("FPS: ", all_result)
	print(f"MFLOPS = {mflops}, MPARAM = {mparam}")
	return all_result, (mflops), (mparam)

	'''

	all_result is a list including n batch_result

	To retrieve the result for each batch:
	- FPS: all_result[batch_index]['rep']['FPS']
		type: float  
	Top-K class for one representive image:all_result[batch_index]['rep']['name']
		type: list including k elements 
	Top-K probability for one representive image: all_result[batch_index]['rep']['prob']
		type: list including k elements 
	Image: all_result[batch_index]['rep']['image']
		type: numpy array <3xHxW>

		'''


#####################################################################
#####################################################################


##################################################
##################################################
# AlphaNetL3 (Improved Sampling – Low Flop) + Hybrid (0.23)
##################################################
##################################################

sys.path.insert(0, './compact_dnn')


def inference_alphanet_l3(batch_size, device):

	use_nas_weights = True
	epochs = 300
	pratio = 0.23

	model = 'sb3'
	lr = 0.01
	training_batchsize = 256
	print("-------------------------")
	print("AlphanetL3")
	print("Batchsize: ", batch_size)

	start_layer = 1
	prune_folder = os.path.join('/home/dso_code/compact_dnn/', 'output_alphanet_prune',
								f'model_{model}', 'pratio_'+str(pratio), 'start_layer_'+str(start_layer))
	sfolder = os.path.join('/home/dso_code/compact_dnn/', 'output_alphanet_hybrid', f'model_{model}', 'pratio_'+str(
		pratio), 'start_layer_'+str(start_layer), f'epochs_{epochs}_lr_{lr}_usenasweights_{use_nas_weights}_batchsize_{training_batchsize}')
	#print(prune_folder)
	#print(sfolder)
	if pratio > 0.001:
		pruned_model = load_pruned_model(
			prune_folder, 'random', device).to(device)
		pruned_model = torch.load(os.path.join(
			prune_folder, 'net.pt'), map_location=device)
		pruned_model = torch.nn.DataParallel(pruned_model)
		hybrid_path = os.path.join(
			sfolder, 'checkpoints', 'ckpttop1_pruning.pth')
		ckpt = torch.load(hybrid_path,  map_location=device)
		pruned_model.load_state_dict(ckpt['net'])
		pruned_model = pruned_model.module
		#print("Loaded from checkpoint %s" % (hybrid_path))
	# else:
		#print("Continue training no pruning")

	pruned_model.eval()

	fps, mflops, mparam = hybrid_nas_infer(
		hybrid_model=pruned_model, batch_size=batch_size, device=device, precision='fp32')
	# print(f'FPS: {fps}')
	print("FPS: ", int(fps))
	print(f"MFLOPS = {mflops}, MPARAM = {mparam}")
	return int(fps), mflops, mparam


##################################################
##################################################
# AlphaNetU7 (Improved Sampling – Uniform Flop) + Hybrid (0.230)
##################################################
##################################################

def inference_alphanet_u7(batch_size, device):

	use_nas_weights = True
	epochs = 300
	pratio = 0.23

	model = 'ue'
	lr = 0.005
	training_batchsize = 128
	print("-------------------------")
	print("AlphanetU7")
	print("Batchsize: ", batch_size)
	
	start_layer = 1
	prune_folder = os.path.join('/home/dso_code/compact_dnn/', 'output_alphanet_prune',
								f'model_{model}', 'pratio_'+str(pratio), 'start_layer_'+str(start_layer))
	sfolder = os.path.join('/home/dso_code/compact_dnn/', 'output_alphanet_hybrid', f'model_{model}', 'pratio_'+str(
		pratio), 'start_layer_'+str(start_layer), f'epochs_{epochs}_lr_{lr}_usenasweights_{use_nas_weights}_batchsize_{training_batchsize}')

	if pratio > 0.001:
		pruned_model = load_pruned_model(
			prune_folder, 'random', device).to(device)
		pruned_model = torch.load(os.path.join(
			prune_folder, 'net.pt'), map_location=device)
		pruned_model = torch.nn.DataParallel(pruned_model)
		hybrid_path = os.path.join(
			sfolder, 'checkpoints', 'ckpttop1_pruning.pth')
		ckpt = torch.load(hybrid_path,  map_location=device)
		pruned_model.load_state_dict(ckpt['net'])
		pruned_model = pruned_model.module
		#print("Loaded from checkpoint %s" % (hybrid_path))
	#else:
		#print("Continue training no pruning")

	pruned_model.eval()

	fps, mflops, mparam = hybrid_nas_infer(
		hybrid_model=pruned_model, batch_size=batch_size, device=device, precision='fp32')
	#print(f'FPS: {fps}')
	print("FPS: ", int(fps))
	print(f"MFLOPS = {mflops}, MPARAM = {mparam}")
	return int(fps), mflops, mparam


#####################################################################
#####################################################################


#####################################################################
#####################################################################
# =====================================================================================
# =====================================================================================
# making things minimalistic
# favicon and page configs
favicon = './assets/favicon.jpg'
st.set_page_config(page_title='SUTD Demo', page_icon=favicon, layout='wide')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)
st.write(
	'<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
hide_streamlit_style = """
		<style>
		#MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		header {visibility: hidden;}
		</style>
		"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# title of the webapp
st.title("SUTD Demo")

# =====================================================================================
# =====================================================================================

# Space out the maps so the first one is 2x the size of the other three
left_side_column, right_side_column = st.columns(2)

with left_side_column:

	# st.image("/media/hung/sidd/demo_v3/assets/right_arrow.png", use_column_width  = True)

	# =====================================================================================
	# =====================================================================================

	selectedSidebar = st.radio(
		"Choose a method",
		("Image Classification", "Object Detection", "Image Segmentation"), horizontal=True)

	# =====================================================================================
	# =====================================================================================

	if selectedSidebar == "Image Classification":

		# =====================================================================================
		# increasing the text size of tab
		font_css = """
		<style>
		button[data-baseweb="tab"] {
		font-size: 20px;
		}
		</style>
		"""
		st.write(font_css, unsafe_allow_html=True)
		# =====================================================================================

		# name of the tabs
		tab_options = ["EfficientNetB0 vs AlphaNetL3", "EfficientNetB3 vs AlphaNetU7"]
		tab_EfficientNetB0_AlphaNetL3, tab_EfficientNetB3_AlphaNetU7 = st.tabs(
			tab_options)

		# batch size options
		batchsize_array = ["128", "64", "32", "16", "8", "4", "2", "1"]

		with tab_EfficientNetB0_AlphaNetL3:
			# st.header("EfficientNetB0")

			# renderData("100 Classes, MFLOPS = 386")

			st.subheader("Please select the inference parameters")

			batchSize = st.selectbox(
				'Choose  the batchsize',
				batchsize_array, key="EfficientNetB0_AlphaNetL3_batchsize")

			modelPrecision = st.radio(
				"",
				('FP32', 'FP16'), horizontal=True, label_visibility="collapsed", key="EfficientNetB0_AlphaNetL3_modelPrecision")

			cpu_gpu = st.radio(
				"",
				('CPU', 'GPU'), horizontal=True, label_visibility="collapsed", key="EfficientNetB0_AlphaNetL3_cpu_gpu")

			# model_name = "EfficientNetB0"

			# st.write('You selected:', batchSize, cpu_gpu, modelPrecision)

			benchmark = st.button("Benchmark", key="EfficientNetB0_AlphaNetL3_benchmark")

			if benchmark:
				fps_array_efficientnet_b0 = []
				fps_array_AlphaNetL3 = []
				time_array = []
				if cpu_gpu == "CPU":
					cpu_gpu = 'cpu'
				else:
					cpu_gpu = 'cuda'
				count = 0
				stop_button = False
				stop_button = st.button("Stop")
				while (stop_button == False):
					with st.spinner("Processing..."):
						count = count + 1
						
						all_result_efficientnet_b0, mflops_efficientnet_b0, mparam_efficientnet_b0 = inference_efficientnet_b0(
							int(batchSize), cpu_gpu)
						fps_efficientnet_b0 = int(all_result_efficientnet_b0)
						fps_array_efficientnet_b0.append(int(fps_efficientnet_b0))

						singaporeTz = pytz.timezone("Asia/Singapore")
						now = datetime.now(singaporeTz)
						current_time = now.strftime("%H:%M:%S")
						time_array.append(current_time)


						
						all_result_AlphaNetL3, mflops_AlphaNetL3, mparam_AlphaNetL3 = inference_alphanet_l3(
							int(batchSize), cpu_gpu)
						fps_AlphaNetL3 = int(all_result_AlphaNetL3)
						fps_array_AlphaNetL3.append(int(fps_AlphaNetL3))


						try:
							info_placeholder_1.empty()
							info_placeholder_2.empty()
							fps_placeholder.empty()
							# fps_placeholder = st.success("EfficientNetB0 and Hybrid FPS on the currrent batch is " + str(int(fps_effb0)) + " and " + str(int(fps_hybrid)) + " respectively.")
						except:
							pass
						
						info_placeholder_1 = st.info(
                            str("EfficientNetB0 - " + "MFLOPS - " + str(mflops_efficientnet_b0) + " , " + "MPARAM - " + str(mparam_efficientnet_b0)))

						info_placeholder_2 = st.info(
                            str("AlphaNetL3 (Improved Sampling – Low Flop) + Hybrid (0.23) - " + "MFLOPS - " + str(mflops_AlphaNetL3) + " , " + "MPARAM - " + str(mparam_AlphaNetL3)))

      

						fps_placeholder = st.success("EfficientNetB0 and AlphaNetL3 FPS on the currrent batch is " + str(int(fps_efficientnet_b0)) + " and " + str(int(fps_AlphaNetL3)) + " respectively.")

						
						
						fps_array_effb0_np = np.array(fps_array_efficientnet_b0)
						fps_array_hybrid_np = np.array(fps_array_AlphaNetL3)
						time_array_np = np.array(time_array)

						df = pd.DataFrame({'EfficientNetB0':fps_array_effb0_np, 'AlphaNetL3':fps_array_hybrid_np, 'time': time_array_np})

						# st.write(df)



						fig = px.line(df, x = 'time', y = ['EfficientNetB0', 'AlphaNetL3'], markers=True, color_discrete_sequence = ["red", "blue"], title = "FPS vs Time")

						fig.update_layout(xaxis_title = 'Time', showlegend = True, yaxis_range = [0,5], legend_title = "Model", title_x = 0.5)
						fig.update_layout(yaxis_title = "FPS", font = dict(family = "Courier New, monospace", size = 14), autosize=False, width=1500,height=1200)
						# autoscale y axis
						fig['layout']['yaxis'].update(autorange = True)

						try:
							placeholder.empty()
						except:
							pass

						with right_side_column:
							placeholder = st.plotly_chart(fig)

		with tab_EfficientNetB3_AlphaNetU7:
			# st.header("EfficientNetB0")

			# renderData("100 Classes, MFLOPS = 386")

			st.subheader("Please select the inference parameters")

			batchSize = st.selectbox(
				'Choose  the batchsize',
				batchsize_array, key="EfficientNetB3_AlphaNetU7_batchsize")

			modelPrecision = st.radio(
				"",
				('FP32', 'FP16'), horizontal=True, label_visibility="collapsed", key="EfficientNetB3_AlphaNetU7_modelPrecision")

			cpu_gpu = st.radio(
				"",
				('CPU', 'GPU'), horizontal=True, label_visibility="collapsed", key="EfficientNetB3_AlphaNetU7_cpu_gpu")

			# model_name = "EfficientNetB0"

			# st.write('You selected:', batchSize, cpu_gpu, modelPrecision)

			benchmark = st.button("Benchmark", key="EfficientNetB3_AlphaNetU7_benchmark")

			if benchmark:
				fps_array_efficientnet_b3 = []
				fps_array_AlphaNetU7 = []
				time_array = []
				if cpu_gpu == "CPU":
					cpu_gpu = 'cpu'
				else:
					cpu_gpu = 'cuda'
				count = 0
				stop_button = False
				stop_button = st.button("Stop")
				while (stop_button == False):
					with st.spinner("Processing..."):
						count = count + 1
						
						all_result_efficientnet_b3, mflops_efficientnet_b3, mparam_efficientnet_b3 = inference_efficientnet_b3(
							int(batchSize), cpu_gpu)
						fps_efficientnet_b3 = int(all_result_efficientnet_b3)
						fps_array_efficientnet_b3.append(int(fps_efficientnet_b3))

						singaporeTz = pytz.timezone("Asia/Singapore")
						now = datetime.now(singaporeTz)
						current_time = now.strftime("%H:%M:%S")
						time_array.append(current_time)


						
						all_result_AlphaNetU7, mflops_AlphaNetU7, mparam_AlphaNetU7 = inference_alphanet_u7(
							int(batchSize), cpu_gpu)
						fps_AlphaNetU7 = int(all_result_AlphaNetU7)
						fps_array_AlphaNetU7.append(int(fps_AlphaNetU7))


						try:
							info_placeholder_1.empty()
							info_placeholder_2.empty()
							fps_placeholder.empty()
						# fps_placeholder = st.success("EfficientNetB0 and Hybrid FPS on the currrent batch is " + str(int(fps_effb0)) + " and " + str(int(fps_hybrid)) + " respectively.")
						except:
							pass
						
						info_placeholder_1 = st.info(
							str("EfficientNetB3 - " + "MFLOPS - " + str(mflops_efficientnet_b3) + " , " + "MPARAM - " + str(mparam_efficientnet_b3)))

						info_placeholder_2 = st.info(
							str("AlphaNetU7 (Improved Sampling – Uniform Flop) + Hybrid (0.230) - " + "MFLOPS - " + str(mflops_AlphaNetU7) + " , " + "MPARAM - " + str(mparam_AlphaNetU7)))

	  
						fps_placeholder = st.success("EfficientNetB3 and AlphaNetU7 FPS on the currrent batch is " + str(int(fps_efficientnet_b3)) + " and " + str(int(fps_AlphaNetU7)) + " respectively.")

						
						
						fps_array_effb3_np = np.array(fps_array_efficientnet_b3)
						fps_array_hybrid_np = np.array(fps_array_AlphaNetU7)
						time_array_np = np.array(time_array)

						df = pd.DataFrame({'EfficientNetB3':fps_array_effb3_np, 'AlphaNetU7':fps_array_hybrid_np, 'time': time_array_np})

						# st.write(df)



						fig = px.line(df, x = 'time', y = ['EfficientNetB3', 'AlphaNetU7'], markers=True, color_discrete_sequence = ["red", "blue"], title = "FPS vs Time")

						fig.update_layout(xaxis_title = 'Time', showlegend = True, yaxis_range = [0,5], legend_title = "Model", title_x = 0.5)
						fig.update_layout(yaxis_title = "FPS", font = dict(family = "Courier New, monospace", size = 14))
						fig.update_layout(
						autosize=False,
						width=1500,
						height=1200,)
						# autoscale y axis
						fig['layout']['yaxis'].update(autorange = True)

						try:
							placeholder.empty()
						except:
							pass

						with right_side_column:
							placeholder = st.plotly_chart(fig)

		
```

```python
# to run the code  
streamlit run demo_v12.py
```
