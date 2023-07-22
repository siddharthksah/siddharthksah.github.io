
> medium-to-markdown@0.0.3 convert
> node index.js https://siddharthksah.medium.com/how-to-build-a-hacky-web-app-that-runs-instance-segmentation-object-detection-and-semantic-62f462b1ddbf

How to build a web app that runs instance segmentation, object detection and semantic segmentation on Nvidia Jetson Orin AGX with low latency and low inference time
====================================================================================================================================================================

[![Siddharth Sah](https://miro.medium.com/v2/resize:fill:88:88/1*RTWoIcWgxb9qaY9qBDHikA.jpeg)

](https://medium.com/?source=post_page-----62f462b1ddbf--------------------------------)

[Siddharth Sah](https://medium.com/?source=post_page-----62f462b1ddbf--------------------------------)

·

[Follow](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fsubscribe%2Fuser%2F1aa18b2cd060&operation=register&redirect=https%3A%2F%2Fsiddharthksah.medium.com%2Fhow-to-build-a-hacky-web-app-that-runs-instance-segmentation-object-detection-and-semantic-62f462b1ddbf&user=Siddharth+Sah&userId=1aa18b2cd060&source=post_page-1aa18b2cd060----62f462b1ddbf---------------------post_header-----------)

13 min read·Feb 5

\--

Listen

Share

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

```
import threading  
import time  
  
def first\_function():  
    """First function to be executed"""  
    for i in range(5):  
        print("Running first function")  
        time.sleep(1)  
  
def second\_function():  
    """Second function to be executed"""  
    for i in range(5):  
        print("Running second function")  
        time.sleep(1)  
  
\# Creating two threads  
thread1 = threading.Thread(target=first\_function)  
thread2 = threading.Thread(target=second\_function)  
  
\# Start both threads  
thread1.start()  
thread2.start()  
  
\# Wait for both threads to complete  
thread1.join()  
thread2.join()  
  
print("Both functions have completed")
```

In the above example, two functions `first_function` and `second_function` are created. Then, two threads are created using `threading.Thread` and each thread is assigned a function to be executed. Finally, both threads are started using the `start()` method and then the main thread waits for both threads to complete using the `join()` method.

Let’s build the webapp now.

```
\# after you have cloned the repo create the conda env and install the requirements  
conda create -n demo python==3.8 -y  
conda activate demo  
pip install -r requirements.txt
```

For starters, let’s split the webapp into two different columns. The left side will be for input from the users and the right side we will show the plots and other information.

```
\# necessary libraries  
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
  
\# importing functions from utils  
from utils.helper import count\_net\_flops\_and\_params, load\_pruned\_model, hybrid\_nas\_infer  
from utils.helper import Customize  
from compact\_dnn.data.data\_loader import build\_data\_loader  
from compact\_dnn.utils.config import setup  
from utils.helper import delete\_temp\_directory, topk2output, tensor2image, evaluate\_efficient  
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
\# EfficientNetB0 #################################  
##################################################  
##################################################  
  
  
def inference\_efficientnet\_b0(batch\_size, device):  
   
 weight\_data\_b0\_placeholder = st.error("Loading EfficientnetB0 weights and data...")  
  
 config = '/home/code/efficientnet/source/configs/imagenet\_100\_b0.yaml'  
  
 config = mlconfig.load(config)  
  
 print("-------------------------")  
 print(config.model.name)  
 print("Batchsize: ", batch\_size)  
  
 model = config.model()  
  
 weight = "/home/code/efficientnet/weights/imagenet100\_b0/best.pth"  
  
   
  
 if weight is not None:  
  checkpoint = torch.load(weight)\['model'\]  
  pretrained\_dict = {key.replace(  
   "module.", ""): value for key, value in checkpoint.items()}  
  model.load\_state\_dict(pretrained\_dict)  
  
 model.to(device)  
  
  
  
 image\_size = 224  # for effb0  
  
 print("Image Size: ", image\_size)  
  
 mflops, mparam = count\_net\_flops\_and\_params(  
  model, data\_shape=(1, 3, image\_size, image\_size))  
  
 # dataset\_b0\_placeholder = st.info("Loading data...")  
  
 rootlabel = "/home/code/efficientnet/dataset/imagenet100/"  
 root = '/home/code/efficientnet/dataset/custom/val'  
  
 valid\_loader = Customize(root, image\_size, False,  
       batch\_size, num\_workers=8)  
 # for x, y in valid\_loader:  
 #     print("#############", x.shape)  
 dataset\_im100 = datasets.ImageFolder(os.path.join(rootlabel, 'val'))  
 class\_names = dataset\_im100.class\_to\_idx  
 class\_names = {y: x for x, y in class\_names.items()}  
  
 # JSON file  
  
 f = open(os.path.join(rootlabel, 'Labels.json'), "r")  
 concrete\_classnames = json.loads(f.read())  
  
 # dataset\_b0\_placeholder.empty()  
 weight\_data\_b0\_placeholder.empty()  
  
 infer\_b0\_placeholder = st.error("Running EfficientnetB0 Inference...")  
  
 all\_result = evaluate\_efficient(  
  model, class\_names, concrete\_classnames, valid\_loader, device)  
 # print(all\_result\[0\]\['rep'\]\['FPS'\])  
 # print(all\_result\[0\]\['rep'\]\['name'\])  
 # print(all\_result\[0\]\['rep'\]\['prob'\])  
 # print(all\_result\[0\]\['rep'\]\['image'\])  
  
 print("FPS: ", all\_result)  
 print(f"MFLOPS = {mflops}, MPARAM = {mparam}")  
  
 infer\_b0\_placeholder.empty()  
  
 return all\_result, (mflops), (mparam)  
  
 '''  
  
 all\_result is a list including n batch\_result  
  
 To retrieve the result for each batch:  
 - FPS: all\_result\[batch\_index\]\['rep'\]\['FPS'\]  
  type: float    
 Top-K class for one representive image:all\_result\[batch\_index\]\['rep'\]\['name'\]  
  type: list including k elements   
 Top-K probability for one representive image: all\_result\[batch\_index\]\['rep'\]\['prob'\]  
  type: list including k elements   
 Image: all\_result\[batch\_index\]\['rep'\]\['image'\]  
  type: numpy array <3xHxW>  
  
  '''  
  
##################################################  
##################################################  
  
  
##################################################  
##################################################  
\# EfficientNetB3 #################################  
##################################################  
##################################################  
  
def inference\_efficientnet\_b3(batch\_size, device):  
   
 weight\_data\_b3\_placeholder = st.error("Loading EfficientnetB3 weights and data...")  
  
 config = '/home/code/efficientnet/source/configs/imagenet\_100\_b3.yaml'  
  
 print("-------------------------")  
 config = mlconfig.load(config)  
  
 model = config.model()  
  
 print(config.model.name)  
 print("Batchsize: ", batch\_size)  
  
 weight = "/home/code/efficientnet/weights/imagenet100\_b3/best.pth"  
  
 if weight is not None:  
  checkpoint = torch.load(weight)\['model'\]  
  pretrained\_dict = {key.replace(  
   "module.", ""): value for key, value in checkpoint.items()}  
  model.load\_state\_dict(pretrained\_dict)  
  
 model.to(device)  
  
 image\_size = 300  # for effb3  
 print("Image Size: ", image\_size)  
  
 mflops, mparam = count\_net\_flops\_and\_params(  
  model, data\_shape=(1, 3, image\_size, image\_size))  
  
 rootlabel = "/home/code/efficientnet/dataset/imagenet100/"  
 root = '/home/code/efficientnet/dataset/custom/val'  
  
 valid\_loader = Customize(root, image\_size, False,  
       batch\_size, num\_workers=8)  
 # for x, y in valid\_loader:  
 #     print("#############", x.shape)  
 dataset\_im100 = datasets.ImageFolder(os.path.join(rootlabel, 'val'))  
 class\_names = dataset\_im100.class\_to\_idx  
 class\_names = {y: x for x, y in class\_names.items()}  
  
 # JSON file  
  
 f = open(os.path.join(rootlabel, 'Labels.json'), "r")  
 concrete\_classnames = json.loads(f.read())  
  
 weight\_data\_b3\_placeholder.empty()  
  
 infer\_b3\_placeholder = st.error("Running EfficientnetB3 Inference...")  
  
 all\_result = evaluate\_efficient(  
  model, class\_names, concrete\_classnames, valid\_loader, device)  
 # print(all\_result\[0\]\['rep'\]\['FPS'\])  
 # print(all\_result\[0\]\['rep'\]\['name'\])  
 # print(all\_result\[0\]\['rep'\]\['prob'\])  
 # print(all\_result\[0\]\['rep'\]\['image'\])  
 print("FPS: ", all\_result)  
 print(f"MFLOPS = {mflops}, MPARAM = {mparam}")  
  
 infer\_b3\_placeholder.empty()  
  
 return all\_result, (mflops), (mparam)  
  
 '''  
  
 all\_result is a list including n batch\_result  
  
 To retrieve the result for each batch:  
 - FPS: all\_result\[batch\_index\]\['rep'\]\['FPS'\]  
  type: float    
 Top-K class for one representive image:all\_result\[batch\_index\]\['rep'\]\['name'\]  
  type: list including k elements   
 Top-K probability for one representive image: all\_result\[batch\_index\]\['rep'\]\['prob'\]  
  type: list including k elements   
 Image: all\_result\[batch\_index\]\['rep'\]\['image'\]  
  type: numpy array <3xHxW>  
  
  '''  
  
  
#####################################################################  
#####################################################################  
  
  
##################################################  
##################################################  
\# AlphaNetL3 (Improved Sampling – Low Flop) + Hybrid (0.23)  
##################################################  
##################################################  
  
sys.path.insert(0, './compact\_dnn')  
  
  
def inference\_alphanet\_l3(batch\_size, device):  
 weight\_data\_l3\_placeholder = st.error("Loading AlphanetL3 weights and data...")  
  
 use\_nas\_weights = True  
 epochs = 300  
 pratio = 0.23  
  
 model = 'sb3'  
 lr = 0.01  
 training\_batchsize = 256  
 print("-------------------------")  
 print("AlphanetL3")  
 print("Batchsize: ", batch\_size)  
  
  
  
 start\_layer = 1  
 prune\_folder = os.path.join('/home/code/compact\_dnn/', 'output\_alphanet\_prune',  
        f'model\_{model}', 'pratio\_'+str(pratio), 'start\_layer\_'+str(start\_layer))  
 sfolder = os.path.join('/home/code/compact\_dnn/', 'output\_alphanet\_hybrid', f'model\_{model}', 'pratio\_'+str(  
  pratio), 'start\_layer\_'+str(start\_layer), f'epochs\_{epochs}\_lr\_{lr}\_usenasweights\_{use\_nas\_weights}\_batchsize\_{training\_batchsize}')  
 #print(prune\_folder)  
 #print(sfolder)  
 if pratio > 0.001:  
  pruned\_model = load\_pruned\_model(  
   prune\_folder, 'random', device).to(device)  
  pruned\_model = torch.load(os.path.join(  
   prune\_folder, 'net.pt'), map\_location=device)  
  pruned\_model = torch.nn.DataParallel(pruned\_model)  
  hybrid\_path = os.path.join(  
   sfolder, 'checkpoints', 'ckpttop1\_pruning.pth')  
  ckpt = torch.load(hybrid\_path,  map\_location=device)  
  pruned\_model.load\_state\_dict(ckpt\['net'\])  
  pruned\_model = pruned\_model.module  
  #print("Loaded from checkpoint %s" % (hybrid\_path))  
 # else:  
  #print("Continue training no pruning")  
  
 weight\_data\_l3\_placeholder.empty()  
  
 infer\_l3\_placeholder = st.error("Running AlphanetL3 inference...")  
  
  
 pruned\_model.eval()  
  
  
 fps, mflops, mparam = hybrid\_nas\_infer(  
  hybrid\_model=pruned\_model, batch\_size=batch\_size, device=device, precision='fp32')  
 # print(f'FPS: {fps}')  
 print("FPS: ", int(fps))  
 print(f"MFLOPS = {mflops}, MPARAM = {mparam}")  
  
 infer\_l3\_placeholder.empty()  
  
 return int(fps), mflops, mparam  
  
  
##################################################  
##################################################  
\# AlphaNetU7 (Improved Sampling – Uniform Flop) + Hybrid (0.230)  
##################################################  
##################################################  
  
def inference\_alphanet\_u7(batch\_size, device):  
   
 weight\_data\_u7\_placeholder = st.error("Loading AlphanetU7 weights and data...")  
  
  
 use\_nas\_weights = True  
 epochs = 300  
 pratio = 0.23  
  
 model = 'ue'  
 lr = 0.005  
 training\_batchsize = 128  
 print("-------------------------")  
 print("AlphanetU7")  
 print("Batchsize: ", batch\_size)  
   
 start\_layer = 1  
 prune\_folder = os.path.join('/home/code/compact\_dnn/', 'output\_alphanet\_prune',  
        f'model\_{model}', 'pratio\_'+str(pratio), 'start\_layer\_'+str(start\_layer))  
 sfolder = os.path.join('/home/code/compact\_dnn/', 'output\_alphanet\_hybrid', f'model\_{model}', 'pratio\_'+str(  
  pratio), 'start\_layer\_'+str(start\_layer), f'epochs\_{epochs}\_lr\_{lr}\_usenasweights\_{use\_nas\_weights}\_batchsize\_{training\_batchsize}')  
  
 if pratio > 0.001:  
  pruned\_model = load\_pruned\_model(  
   prune\_folder, 'random', device).to(device)  
  pruned\_model = torch.load(os.path.join(  
   prune\_folder, 'net.pt'), map\_location=device)  
  pruned\_model = torch.nn.DataParallel(pruned\_model)  
  hybrid\_path = os.path.join(  
   sfolder, 'checkpoints', 'ckpttop1\_pruning.pth')  
  ckpt = torch.load(hybrid\_path,  map\_location=device)  
  pruned\_model.load\_state\_dict(ckpt\['net'\])  
  pruned\_model = pruned\_model.module  
  #print("Loaded from checkpoint %s" % (hybrid\_path))  
 #else:  
  #print("Continue training no pruning")  
  
 weight\_data\_u7\_placeholder.empty()  
  
 infer\_u7\_placeholder = st.error("Running AlphanetU7 inference...")  
  
  
 pruned\_model.eval()  
  
 fps, mflops, mparam = hybrid\_nas\_infer(  
  hybrid\_model=pruned\_model, batch\_size=batch\_size, device=device, precision='fp32')  
 #print(f'FPS: {fps}')  
 print("FPS: ", int(fps))  
 print(f"MFLOPS = {mflops}, MPARAM = {mparam}")  
  
 infer\_u7\_placeholder.empty()  
  
 return int(fps), mflops, mparam  
  
  
#####################################################################  
#####################################################################  
  
  
#####################################################################  
#####################################################################  
\# =====================================================================================  
\# =====================================================================================  
\# making things minimalistic  
\# favicon and page configs  
favicon = './assets/favicon.jpg'  
st.set\_page\_config(page\_title='Project ECLIPSIS Demo: Extremely Compact DNNs for Edge Intelligence', page\_icon=favicon, layout='wide')  
\# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)  
st.write(  
 '<style>div.block-container{padding-top:0rem;}</style>', unsafe\_allow\_html=True)  
hide\_streamlit\_style = """  
  <style>  
  #MainMenu {visibility: hidden;}  
  footer {visibility: hidden;}  
  header {visibility: hidden;}  
  </style>  
  """  
st.markdown(hide\_streamlit\_style, unsafe\_allow\_html=True)  
  
st.image("./assets/header.jpg")  
  
\# title of the webapp  
#original\_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Project ECLIPSIS Demo: Extremely Compact DNNs for Edge Intelligence</p>'  
#st.markdown(original\_title, unsafe\_allow\_html=True)  
#st.title("Project ECLIPSIS Demo: Extremely Compact DNNs for Edge Intelligence")  
  
\# =====================================================================================  
\# =====================================================================================  
  
\# Space out the maps so the first one is 2x the size of the other three  
left\_side\_column, right\_side\_column = st.columns(2)  
  
with left\_side\_column:  
  
 # st.image("/media/hung/sidd/demo\_v3/assets/right\_arrow.png", use\_column\_width  = True)  
  
 # =====================================================================================  
 # =====================================================================================  
  
 selectedSidebar = st.radio(  
  "Choose a method",  
  ("Image Classification", "Image Segmentation"), horizontal=True)  
   
 # selectedSidebar = "Image Classification"  
 # =====================================================================================  
 # =====================================================================================  
 st.warning("Image Classification")  
 if selectedSidebar == "Image Classification":  
  
  # =====================================================================================  
  # increasing the text size of tab  
  font\_css = """  
  <style>  
  button\[data-baseweb="tab"\] {  
  font-size: 14px;  
  }  
  </style>  
  """  
  st.write(font\_css, unsafe\_allow\_html=True)  
  # =====================================================================================  
  
  # name of the tabs  
  tab\_options = \["EfficientNetB0 vs AlphaNetL3", "EfficientNetB3 vs AlphaNetU7"\]  
  tab\_EfficientNetB0\_AlphaNetL3, tab\_EfficientNetB3\_AlphaNetU7 = st.tabs(  
   tab\_options)  
  
  # batch size options  
  batchsize\_array = \["128", "64", "32", "16", "8", "4", "2", "1"\]  
  
  with tab\_EfficientNetB0\_AlphaNetL3:  
   # st.header("EfficientNetB0")  
  
   # renderData("100 Classes, MFLOPS = 386")  
  
   #st.write("Please select the inference parameters")  
  
   batchSize = st.selectbox(  
    'Choose  the batchsize',  
    batchsize\_array, key="EfficientNetB0\_AlphaNetL3\_batchsize")  
  
   modelPrecision = "FP32"  
   #modelPrecision = st.radio(  
   # "",  
   # ('FP32', 'FP16'), horizontal=True, label\_visibility="collapsed", key="EfficientNetB0\_AlphaNetL3\_modelPrecision")  
  
   cpu\_gpu = "GPU"  
   #cpu\_gpu = st.radio(  
   # "",  
   # ('CPU', 'GPU'), horizontal=True, label\_visibility="collapsed", key="EfficientNetB0\_AlphaNetL3\_cpu\_gpu")  
  
   # model\_name = "EfficientNetB0"  
  
   # st.write('You selected:', batchSize, cpu\_gpu, modelPrecision)  
     
   st.success("Running with FP32 Precision on GPU")  
  
   benchmark = st.button("Benchmark", key="EfficientNetB0\_AlphaNetL3\_benchmark")  
  
   if benchmark:  
    fps\_array\_efficientnet\_b0 = \[\]  
    fps\_array\_AlphaNetL3 = \[\]  
    time\_array = \[\]  
    if cpu\_gpu == "CPU":  
     cpu\_gpu = 'cpu'  
    else:  
     cpu\_gpu = 'cuda'  
    count = 0  
    stop\_button = False  
    stop\_button = st.button("Stop")  
    while (stop\_button == False):  
     with st.spinner("Processing..."):  
      count = count + 1  
        
      all\_result\_efficientnet\_b0, mflops\_efficientnet\_b0, mparam\_efficientnet\_b0 = inference\_efficientnet\_b0(  
       int(batchSize), cpu\_gpu)  
      fps\_efficientnet\_b0 = int(all\_result\_efficientnet\_b0)  
      fps\_array\_efficientnet\_b0.append(int(fps\_efficientnet\_b0))  
  
      singaporeTz = pytz.timezone("Asia/Singapore")  
      now = datetime.now(singaporeTz)  
      current\_time = now.strftime("%H:%M:%S")  
      time\_array.append(current\_time)  
  
  
        
      all\_result\_AlphaNetL3, mflops\_AlphaNetL3, mparam\_AlphaNetL3 = inference\_alphanet\_l3(  
       int(batchSize), cpu\_gpu)  
      fps\_AlphaNetL3 = int(all\_result\_AlphaNetL3)  
      fps\_array\_AlphaNetL3.append(int(fps\_AlphaNetL3))  
  
  
      try:  
       info\_placeholder\_1.empty()  
       info\_placeholder\_2.empty()  
       fps\_placeholder.empty()  
       # fps\_placeholder = st.success("EfficientNetB0 and Hybrid FPS on the currrent batch is " + str(int(fps\_effb0)) + " and " + str(int(fps\_hybrid)) + " respectively.")  
      except:  
       pass  
        
      info\_placeholder\_1 = st.info(  
       str("EfficientNetB0 - " + "MFLOPS - " + str(mflops\_efficientnet\_b0) + " , " + "MPARAM - " + str(mparam\_efficientnet\_b0)))  
  
      info\_placeholder\_2 = st.info(  
       str("AlphaNetL3 (Improved Sampling – Low Flop) + Hybrid (0.23) - " + "MFLOPS - " + str(mflops\_AlphaNetL3) + " , " + "MPARAM - " + str(mparam\_AlphaNetL3)))  
  
  
  
      fps\_placeholder = st.success("EfficientNetB0 and AlphaNetL3 FPS on the currrent batch is " + str(int(fps\_efficientnet\_b0)) + " and " + str(int(fps\_AlphaNetL3)) + " respectively.")  
  
        
        
      fps\_array\_effb0\_np = np.array(fps\_array\_efficientnet\_b0)  
      fps\_array\_hybrid\_np = np.array(fps\_array\_AlphaNetL3)  
      time\_array\_np = np.array(time\_array)  
  
      df = pd.DataFrame({'EfficientNetB0':fps\_array\_effb0\_np, 'AlphaNetL3':fps\_array\_hybrid\_np, 'time': time\_array\_np})  
  
      # st.write(df)  
  
  
  
      fig = px.line(df, x = 'time', y = \['EfficientNetB0', 'AlphaNetL3'\], markers=True, color\_discrete\_sequence = \["red", "blue"\], title = "FPS vs Time")  
  
      fig.update\_layout(xaxis\_title = 'Time', showlegend = True, yaxis\_range = \[0,5\], legend\_title = "Model", title\_x = 0.5)  
      fig.update\_layout(yaxis\_title = "FPS", font = dict(family = "Courier New, monospace", size = 14), autosize=False, width=1500,height=1200)  
      # autoscale y axis  
      fig\['layout'\]\['yaxis'\].update(autorange = True)  
      images\_processed = 100\*int(batchSize)\*count  
      images\_processed\_string = "Total number of images inferred: " + str(images\_processed)  
      #st.write(images\_processed)  
  
      try:  
       placeholder.empty()  
       processing\_placeholder.empty()  
      except:  
       pass  
  
      with right\_side\_column:  
       placeholder = st.plotly\_chart(fig)  
       #processing\_placeholder = st.success(images\_processed\_string)  
       processing\_placeholder = st.metric(label="Total number of images inferred", value=str(images\_processed), delta=str(int(batchSize)\*100))  
  
  
  with tab\_EfficientNetB3\_AlphaNetU7:  
   # st.header("EfficientNetB0")  
  
   # renderData("100 Classes, MFLOPS = 386")  
  
   #st.write("Please select the inference parameters")  
  
   batchSize = st.selectbox(  
    'Choose  the batchsize',  
    batchsize\_array, key="EfficientNetB3\_AlphaNetU7\_batchsize")  
   modelPrecision = "FP32"  
   #modelPrecision = st.radio(  
   # "",  
   # ('FP32', 'FP16'), horizontal=True, label\_visibility="collapsed", key="EfficientNetB3\_AlphaNetU7\_modelPrecision")  
   cpu\_gpu = "GPU"  
   #cpu\_gpu = st.radio(  
   # "",  
   # ('CPU', 'GPU'), horizontal=True, label\_visibility="collapsed", key="EfficientNetB3\_AlphaNetU7\_cpu\_gpu")  
  
   # model\_name = "EfficientNetB0"  
  
   # st.write('You selected:', batchSize, cpu\_gpu, modelPrecision)  
   st.success("Running with FP32 Precision on GPU")  
  
   benchmark = st.button("Benchmark", key="EfficientNetB3\_AlphaNetU7\_benchmark")  
  
   if benchmark:  
    fps\_array\_efficientnet\_b3 = \[\]  
    fps\_array\_AlphaNetU7 = \[\]  
    time\_array = \[\]  
    if cpu\_gpu == "CPU":  
     cpu\_gpu = 'cpu'  
    else:  
     cpu\_gpu = 'cuda'  
    count = 0  
    stop\_button = False  
    stop\_button = st.button("Stop")  
    while (stop\_button == False):  
     with st.spinner("Processing..."):  
      count = count + 1  
        
      all\_result\_efficientnet\_b3, mflops\_efficientnet\_b3, mparam\_efficientnet\_b3 = inference\_efficientnet\_b3(  
       int(batchSize), cpu\_gpu)  
      fps\_efficientnet\_b3 = int(all\_result\_efficientnet\_b3)  
      fps\_array\_efficientnet\_b3.append(int(fps\_efficientnet\_b3))  
  
      singaporeTz = pytz.timezone("Asia/Singapore")  
      now = datetime.now(singaporeTz)  
      current\_time = now.strftime("%H:%M:%S")  
      time\_array.append(current\_time)  
  
  
        
      all\_result\_AlphaNetU7, mflops\_AlphaNetU7, mparam\_AlphaNetU7 = inference\_alphanet\_u7(  
       int(batchSize), cpu\_gpu)  
      fps\_AlphaNetU7 = int(all\_result\_AlphaNetU7)  
      fps\_array\_AlphaNetU7.append(int(fps\_AlphaNetU7))  
  
  
      try:  
       info\_placeholder\_1.empty()  
       info\_placeholder\_2.empty()  
       fps\_placeholder.empty()  
      # fps\_placeholder = st.success("EfficientNetB0 and Hybrid FPS on the currrent batch is " + str(int(fps\_effb0)) + " and " + str(int(fps\_hybrid)) + " respectively.")  
      except:  
       pass  
        
      info\_placeholder\_1 = st.info(  
       str("EfficientNetB3 - " + "MFLOPS - " + str(mflops\_efficientnet\_b3) + " , " + "MPARAM - " + str(mparam\_efficientnet\_b3)))  
  
      info\_placeholder\_2 = st.info(  
       str("AlphaNetU7 (Improved Sampling – Uniform Flop) + Hybrid (0.230) - " + "MFLOPS - " + str(mflops\_AlphaNetU7) + " , " + "MPARAM - " + str(mparam\_AlphaNetU7)))  
  
  
      fps\_placeholder = st.success("EfficientNetB3 and AlphaNetU7 FPS on the currrent batch is " + str(int(fps\_efficientnet\_b3)) + " and " + str(int(fps\_AlphaNetU7)) + " respectively.")  
  
        
        
      fps\_array\_effb3\_np = np.array(fps\_array\_efficientnet\_b3)  
      fps\_array\_hybrid\_np = np.array(fps\_array\_AlphaNetU7)  
      time\_array\_np = np.array(time\_array)  
  
      df = pd.DataFrame({'EfficientNetB3':fps\_array\_effb3\_np, 'AlphaNetU7':fps\_array\_hybrid\_np, 'time': time\_array\_np})  
  
      # st.write(df)  
  
  
  
      fig = px.line(df, x = 'time', y = \['EfficientNetB3', 'AlphaNetU7'\], markers=True, color\_discrete\_sequence = \["red", "blue"\], title = "FPS vs Time")  
  
      fig.update\_layout(xaxis\_title = 'Time', showlegend = True, yaxis\_range = \[0,5\], legend\_title = "Model", title\_x = 0.5)  
      fig.update\_layout(yaxis\_title = "FPS", font = dict(family = "Courier New, monospace", size = 14))  
      fig.update\_layout(  
      autosize=False,  
      width=1500,  
      height=1200,)  
      # autoscale y axis  
      fig\['layout'\]\['yaxis'\].update(autorange = True)  
      images\_processed = 100\*int(batchSize)\*count  
      images\_processed\_string = "Total number of images inferred: " + str(images\_processed)  
        
      try:  
       placeholder.empty()  
       processing\_placeholder.empty()  
      except:  
       pass  
  
      with right\_side\_column:  
       placeholder = st.plotly\_chart(fig)  
       processing\_placeholder = st.metric(label="Total number of images inferred", value=str(images\_processed), delta=str(int(batchSize)\*100))  
  
  
    
 if selectedSidebar == "Image Segmentation":  
  st.write("Test")  
  # st.write("Image Segmentation")
``````
\# to run the code  
streamlit run demo\_v12.py
```
