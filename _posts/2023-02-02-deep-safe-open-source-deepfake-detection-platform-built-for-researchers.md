---
title: 'DeepSafe: Open source deepfake detection platform built for Researchers'
date: 2023-02-02
permalink: /posts/2023/02/deep-safe-open-source-deepfake-detection-platform-built-for-researchers/
tags:
  - Machine Learning
  - CICD
  - Project Structure
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
            Welcome to the realm of "DeepSafe: Open source deepfake detection platform built for Researchers," where we present an advanced and meticulously crafted solution to combat the growing menace of deepfake technology. This article unveils the technical prowess and functionalities of DeepSafe, empowering researchers with a powerful tool to discern and address the challenges posed by deceptive AI-generated media.
        </p>
    </div>
    <div class="intro-image">
        <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/posts/navigating-the-maze-streamlined-project-structure-for-data-science-and-everything-around-it.jpeg?raw=true">
        <p class="image-caption"><em>Image generated using text-to-image model by Adobe</em></p>
    </div>
</div> 

    
```python
https://github.com/siddharthksah/DeepSafe  
Hosted here — http://deepsafe.ml/ (If you want a demo and this is down please feel free to contact me)
```
    
# Background

DeepFake detection is important especially as the video content on the
internet is getting more and more common. With the rise of influencer
marketing on TikTok and Instagram Reels, creation of short video content is at
an all-time high. In fact, companies like TikTok have beaten giants like
Google in this sector. As these video contents are consumed on a variety of
different platforms from smartphone to desktop, the deepfake detection
solution needs to run on all these platforms seamlessly.

To understand the use case and practicality of deepfake detection on these
devices, I explored multiple solutions in the form of different technologies
and different use cases to understand which solution would be most practical.
To quantify the results, a survey was carried out where users were given to
try out all these solutions to deepfake detection and their experience was
noted and studied. This type of study is unique and has not been carried with
multiple solutions in the literature studied. This thesis aims to quantify the
best proposed method to detect deepfake videos in the day to day social media
content consumption. The solutions built and studied have been listed below.

  * DeepSafeWebApp/DeepSafe Mobile App
  * DeepSafe Chrome Extension with client side processing
  * DeepSafe Chrome Extension with server side processing

# Why DeepSafe?

There is a need to keep in check on the fake information spread on social
media more than ever. Fake news detection based on text data is a very well
studied subject both in academia and industry. Most of the tools we use on the
internet for texting come equipped for some sort of fake news detector.

There has been some work on creating platforms for DeepFake detection.
DeepFakeo- meter created by Li et al was published in 2021. They incorporated
a few of the existing detectors and ran inferences on images and videos on
their webapp. Although, there is a lot of limitations within the webapp
itself. It only supports uploading local files, no URLs. This renders the
whole idea of using deepfake detection for social media unusable. There are
also hard limits on the file size and no backend API is provided. DeepFake-o-
meter has been only created for end users primarily. The features on API or
adding custom detectors do not exist.

DeepSafe has been built with platform design in the core. It aims to provide
additional custom features which can be incorporated on the platform right
away. It comes with sanity checks and adding custom detection methods with
minimal changes right out of the box. A platform like this is important
because of the variety of generation and detection methods in academia being
studied right now. An absence of a unified platform to use SOTA DeepFake
Detectors is why DeepSafe was built.

Some of the deepfake creation techniques work well on edge devices. There are
usually low res videos created using FaceSwap. They are usually harmless and
created for memes. These apps run on mobile devices and need no technical
knowledge to create these videos. As it has become handy the increase in the
number of videos on being shared as memes has gone up exponentially. These
creation methods are dynamic which means it gets more real every year.
Currently, these videos are evident “enough” to look fake hence it has no
intentional malicious use. But it is not very difficult to create high res
ones, hence platforms like DeepSafe which can run right on your mobile phone
are as relevant as ever

Companies like Deepware and Sensity aim to solve the deepfake detection issue
as well. These companies are mostly closed source with some sample open source
models. The platform design and code is closed source. Their target audience
are SME and direct users. These are designed on the top of the front end and
help these companies combat spread of fake media on their platform. The target
audience is different. Deep- Safe is built by the researchers for the
researchers solving the pain points of carrying out sustainable research in
the deepfake detection domain. Building better detectors needs to instil
knowledge from the previous generation ones. To create a better deepfake
detector, researchers need to keep track and understand the previous built
detectors. Every few months there is a new detector and generator solving the
issues from the previous generations rendering the detectors useful in most
cases. To keep a track of this highly dynamic research domain DeepSafe has
been created.

DeepSafe not only is an aggregator of detection models but it also creates a
dataset of all the potentially harmful circulated DF images and videos. An
absence of a platform which can authenticate or flag a video is important to
keep the trustworthiness in media alive. DeepSafe incorporates a 2 min quick
guide to detect potential deepfakes in a glance based on the research until
now. Blockchain and media file signatures can be integrated into DeepSafe to
keep track of image and videos and its source. A standard way of detection is
running detection models on a media file and outputting the probability.
DeepSafe uses an extra step to save writing repetitive tests and provides a
dashboard to keep track of all the experiments. Rather than running a
detection model first, DeepSafe first looks for URL matching in the database.
Image matching and Video fingerprint matching is also an important step to it.
Video finger print is still in development. Currently, DeepSafe incorporates
URL and image matching only. Although, by default this setup is toggled off.
DeepSafe has a dynamic database, which means if we detect a deepfake we add it
to our library of deepfakes after running it through other detectors to reduce
the TN and FP.

There are a few competitors which do something similar. But they have their
limitations, like 1. Closed source 2. Only supports uploaded image/video and
only few URLs 3. All of them are static and use their proprietary detection
algorithms 4. No open-source database

![](2023-02-02-deep-safe-open-source-deepfake-detection-platform-built-for-researchers/1*GUaS7VJ0ZvaVhHImNlV7jQ.png?raw=True)

Key features of DeepSafe

# DeepSafe WebApp

The DeepSafe WebApp provides a web UI which auto adapts to mobile and desktop
views. The UI allows users to paste URLs of images and videos which they want
to check the authenticity of. The URL supports the majority of the common
image and video formats on the internet. DeepSafe downloads these media files
and converts them into MP4 for video files and JPG for images. This ensures
the URLs of video or images with non standard formats also work. Saving media
in a single standard format ensures no failures in detection caused by file
formats and support of different detectors on these formats.

The media downloader from the URL is based on the yt-dlp package. This
supports the majority of the public videos and websites. A list of supported
websites is provided in the appendix.

Along with the URL, users can also directly upload files from their local
system. The image formats supported for uploading in the latest release as of
July’22 are JPG, PNG, JPEG and video formats are MP4 and MOV. DeepSafe
converts the uploaded files into MP4 for video files and JPG for images. This
ensures the videos or images with non standard formats also work. Saving media
in a single standard format ensures no failures in detection caused by file
formats and support of different detectors on these formats.

Examples on DeepSafe WebApp

DeepSafe includes the examples of the most common SFW deepfake videos on the
internet. These videos can be played right on the DeepSafe platform without
leaving it. The idea behind this is to educate people what high quality
deepfakes might look like.

![](local/1*vFA79jR_zPFyXmN9YjvLFg.png)

An illustration of DeepSafe WebApp Main Page

![](local/1*RD0dSO6TS2-PaaXe4Z9TtA.png)

DeepSafe WebApp About section — contains information about detectors
incorporated onto the platform

![](local/1*6bTOqqLd1m3tn9wOT5luFA.png)

A quick guide to learn to spot DeepFakes

![](local/1*3AUbcOfNhrK09Rz4tCRN3A.png)

An illustration of how Deep Learning works in practice

DeepSafe Mobile App

DeepSafe mobile app has both the versions for ios and android. The processing
in done on the server side and not on the mobile phones hence the user
experience related to inference time remains the same.

![](local/1*T9IZ7ThF85RYAios3_5ieg.png)

Mobile Apps for iOS and Android

![](local/1*I8t5j5GtoFcVxNAZMei_EA.png)

Usage of URLs and local uploads

## DeepSafe Chrome Extension with client side processing

The first version of DeepSafe Chrome Extension was based on processing media
files right in the browser. While this methodology ensures significantly lower
server costs for the host platform but it comes with a great expense of user
experience. DeepFake detection especially on video files can be
computationally expensive which if run on a slower system can even freeze.
Moreover, each video needs to be processed every time a user wants to check.
If the detection is carried right onto the server after the video or image is
uploaded onto the platform it ensures the check happens only once. Also, it
can flag such videos without even reaching the public and spreading potential
misinfomation. Processing data in user’s browser also solves the issue of data
privacy and handles any data storage cost. In case of video files when the
file size is bigger compared to image files, transferring the data between
client and server takes time effectively increasing the total latency.

Building a detection system should not rely on the client side for processing
unless the total inference is done under a second on a standard edge device.
As DeepFake detection, especially on video files can be computationally
expensive, the processing being done on the server side makes the user
experience better. Although it comes at the cost of data transfer in case the
user wants to upload their own custom video file. As that particular case
wouldn’t be the user browsing, not having a real-time doesn’t affect the user
experience.

The v1 of DeepSafe chrome extension focussed on doing the inference on images.
The user experience was also limited to it. No inference was done on video
files in this case.

![](local/1*UAbzRSzzYl-L7wGficvFgQ.png)

An illustration of how the first version of DeepFake chrome extension work

## DeepSafe Chrome Extension with server side processing

As having a chrome extension continuously running in the background can be
computationally expensive, v2 of the extension focussed on understanding this
current limitation.

If we look at the social media content, the majority of them are media files.
With the rise of TikTok and other short videos platforms, consumption of media
files has leaned towards short video clips. The DeepSafe detection system
architecture is specifically designed to support these video files and their
URL. V2 chrome extension opens the video in a new tab of the browser and
processes it to get the deepfake probability.

## URL Matching

There are multiple ways to reduce the processing time and computational
expense during detection. The classic ones are reducing or downsizing the
image and video files to lower resolution. This reduces the input layer size
and hence reduces the total computation. Another method is model quantization
and pruning. This method essentially reduces the model size removing weights
by making them zero. Although it does impact the accuracy there is a trade off
between inference time and accuracy but selecting the right parameters can
significantly reduce the processing time hence increasing the user experience.

The entire discussion around processing time is the key because the user
experience significantly improves if the inferences are as real time as
possible. Inference shouldn’t be the bottleneck in the ideal situation.

One of the interesting ways to reduce computation is using URL matching. The
idea is to store the URLs of the image or video which are flagged as deepfake
by DeepSafe. If this flag is in infact true we can store it in our database.
Hence before running an inference of a new media we can search for it in the
database and return back the flag. Although this seems more like an ad hoc
solution. The flag can be verified by crowd sourcing or using other deepfake
detection models. All the detectors can test this media file and output a
probability distribution. This probability distribution will increase the
confidence score of the flag and in case it is higher than a preset threshold
we can flag or unflag.

URL matching is especially important for video files. It reduces the
computational cost by hundreds of times if not thousands. Additionally, saving
URLs cheap and well practiced.

## Media similarity

By definition, similarity is a hazy idea. It works well when two people talk,
but it is not the simplest thing for a deterministic computer programme to
cope with. If the aim is to analyse certain characteristics of a picture,
creating and training a model for that purpose works, such as a standard image
classification model. However, if the “similarity” requirements are visual but
not well defined, or if there is a lack of finely labelled data for training,
picture similarity using deep learning may help. Deep learning paves the way
for reliable quantification of picture similarity, allowing for the automation
of even ill-defined tasks.

The most apparent use is probably reverse searching. Reverse searching allows
us to start with the content and get the keywords linked with it. Deep
learning similarity approaches are also at the heart of some of the most
effective methods for landmark identification, which may help us recall where
to focus in photos. Picture similarity may be used for the data clustering if
we want to keep things ordered. This enables us to use a mix of explicit
information, such as clothing type, and visual cues learnt implicitly by a
deep learning model.

However, unlike a traditional hash function, the compressed representations
should be comparable such that the degree of similarity between their
respective items may be plainly indicated.

“Cosine similarity” is the operation that is typically used to assess how
close two things are when it comes to similarity. When we read that two photos
are 90% similar, it most likely implies that their compressed representation’s
cosine similarity is equal to 0.90, or that they are 25 degrees apart. The
similarity of two vectors in an inner product space is measured by cosine
similarity. It is calculated by taking the cosine of the angle between two
vectors and determining if two vectors are heading in the same general
direction. It is often used in text analysis to assess document similarity.
Cosine similarity is a mathematical measure of the cosine of the angle between
two vectors projected in a multidimensional space.

When plotted on a multi-dimensional space, where each dimension corresponds to
a word in the text, the cosine similarity captures the direction (the angle)
but not the amplitude. The cosine similarity is helpful because, even if two
comparable papers are separated by the Euclidean distance due to size (for
example, the word “cricket” occurred 50 times in one document and 10 times in
another), they may still have a lesser angle between them. The greater the
resemblance, the smaller the angle.

Using image and video similarity techniques can also help us look at the
existing database and flag if an image or video is deepfake without running
inferences on them, saving computation.

 **Front-end and Back-end**

DeepSafe’s front end is based on Streamlit. DeepSafe automatically detects the
file type from the URL or the uploaded file and presents only the relevant
detection methods.

 **DeepSafe Dashboard**

DeepSafe has a built-in dashboard which provides all the information and plots
of the experiments run. This saves all the information about inference on
different images and videos. Also, it saves a local copy of these files in
case it is detected as DeepFake which can be used in retraining your detection
models. The dashboard contains the following information. Number of
Inferences, Timestamp, Average DF probability, File Type, File Size in MB,
Total processing time in seconds.

 **Streamlit**

Streamlit is an open-source Python application framework. It expedites the
creation of web applications for data science and machine learning. It is
interoperable with important Python libraries, including scikit-learn,
PyTorch, Keras, NumPy, Matplotlib and pandas, among others.

Streamlit does not need callbacks since widgets are viewed as variables.
Caching data simplifies and accelerates calculation processes. Streamlit
monitors for modifications to the connected Git repository and deploys the
application automatically in the shared link. Streamlit streamlines the
interactive cycle of coding and visualisation in a web application.

 **Project Structure**

The procedure to integrate new models onto the platform locally is simple.
Unlike other methods, we do not need to change any parameters into the
original model what- soever. The idea is to have an extra .py file which runs
the inference using the command line right inside a python file. This is
achieved by calling the inference code in a python pseudo terminal. The
approach standardised both the inputs and outputs. Ideally the output should
be a text file which the probability of deepfake in the root directory. The
exception handling comes by default out of the box which includes readable
error codes which point out exactly where the code broke during the process
which makes it easier to debug. Also, the sanity check which comes out of the
box when run checks for all the potential bugs and gives a report of the
entire pipeline with just one click. The version control is handled such that
all the models have their own environment which provides a very smooth
integration of models over years with different versions of libraries.

 **How does DeepSafe help DeepFake researchers?**

Since the last few years we have seen a rise in the amount of deepfake videos
and photos on the internet. Some of these videos are created with malicious
purposes which need to be detected/removed from the platform. As the new
videos make the old detectors obsolete it is important to learn and understand
how the old detectors work in order to create the new ones. DeepSafe comes
handy here. Any researcher can create a local copy of DeepSafe and install the
dependencies in just a few commands. These commands have been tested and
customised anc verified for Windows, Mac and Ubuntu OS. Majority of these
detectors are not in active development and do not come with the installation
requirements which makes it tricky and at times next to impossible to create
the right environment to test. Also, over time these detectors have different
versions of python and associated libraries which makes it troublesome to test
all at once. DeepSafe saves weeks of these trouble.

 **How to add your own DeepFake Detector?**

DeepSafe allows a simple and intuitive way of adding any custom detector and
comes with out of the box code to test it against other state of the art
detectors.

  1.  **Folder Structure**
  2.  **Meta Data**

Metadata includes the licence, the URL to the project, type of detector. An
example of Meta data is below

Boken, https://github.com/beibuwandeluori/DeeperForensicsChallengeSolution,
MIT Licence

  1.  **Demo.py**

This is the most important aspect which connects any deepfake detector to
DeeSafe. The demo.py calls the model and saves the inference result in a text
file in the model folder. Note that this is saved in the respective model
folder and not the root.

  1.  **requirements.txt**

This should contain all the relevant libraries and their versions needed The
folder name should be “detectorname-image” or “detectorname-video”, depending
on whether it is an image or video detector.

  1.  **Out of the box features**

Run inference on any dataset with all the detectors. This will also provide an
estimated time of completion. Drag and drop folders in the main repo without
any extra configuration.

![](local/1*BZjX_86IQvaS0pXA7qmq2A.png)

info-graphics on the folder structure of the DeepSafe Models

 **Sanity Check**

Checks all the required model settings and runs inference with all the models
on test images and videos to check if everything is working fine.

 **DockerFile**

Docker is essentially a very light-weight virtual machine that contains all of
the applications and dependencies required to execute your programme. When we
set up a large application with a UI, database, and many microservices, we
must complete a setup, and each of these services need some type of software
to function, such as JAVA. And setting up and installing everything properly
on the new server would most likely take days.

However, most of the time, versions do not match and other issues arise,
necessitating extensive debugging. So, what Dockers does is construct an image
and describe what is required for our software. So we’re just saying these are
the dependencies, this is where our programme is coming from, and then we need
to construct a container with our software, which we’re simply fetching from
the Docker registry. And a container containing a software will be running,
ready to boot in a few seconds. So we merely download and start in a few
seconds, and everybody who requests that image will have the exact same
configuration. The biggest benefit of Dockers is the ability to have the exact
same configuration everywhere and to swiftly bootstrap the whole
infrastructure. Containers may also be built quite rapidly.

Docker is a great technology to use if we want to create a single application
that can run and operate identically in any environment, laptop, or public
cloud instance, owing to its convergent virtualized packages known as
containers.

A Dockerfile is a basic text file that contains instructions for creating
Docker images. The benefit of using a Dockerfile over just saving the binary
image (or a snapshot/template in other virtualization systems) is that
automated builds guarantee we always have the most recent version. This is a
good thing from a security standpoint, as you want to guarantee that no
susceptible software is installed.

 **How are the examples on DeepSafe WebApp collected?**

DeepSafe web app contains Safe for Work(SFW) deepfake videos for educational
purposes. These videos have been collected through web scraping on YouTube.

 **Current versions of DeepSafe WebApp**

The DeepSafe webapp has 2 versions for the final prototype. The first one,
hosted on Google Cloud Run, comes with limited features and can be used on the
web without any installation whatsoever. This version has been built for end
users who want to test if the video or image they are watching is DeepFake or
not. The second version of the webapp is built for the Researchers who can use
it to incorporate their custom detectors and test it against custom datasets.
The features of the later version have been listed in the previous paragraphs.
The first version of DeepSafe mentioned is Proof-of-concept work only. The
thesis is based on the second version of it.

 **DeepSafe API**

There are multiple options when using the DeepSafe API. Run inference on an
entire folder across all the detectors out of the box. Run inference on an
entire folder with specific detectors only. The API is available on GitHub and
is also open source under MIT Licence.

 **API Design**

Pass images in base64 encoding with a number which represents which SOTA
algorithm to use for DeepFake Detection get json output with the probability
float value.

API is a collection of procedures, protocols, and tools used to develop
software applications. An API essentially defines how software components
should communicate. They are also used to programme components of graphical
user interfaces (GUIs). A good API facilitates programme development by
offering all the necessary components. Modern APIs conform to developer-
friendly, widely accessible, and commonly understood standards (usually HTTP
and REST). They are handled as products rather than code. They are intended
for consumption by certain audiences (e.g., mobile developers), are
documented, and are versioned so that users may have certain maintenance and
lifetime expectations. They have a much greater discipline for security and
governance, as well as performance and scale monitoring and management, since
they are much more standardised. Modern APIs have their own software
development lifecycle (SDLC) consisting of designing, testing, constructing,
maintaining, and versioning.

Flask is a popular micro web framework for developing Python APIs. It is a
basic but powerful web framework built for rapid and easy application
development, with the capacity to expand to large projects. “Micro” does not
imply that your whole web application must be contained inside a single Python
file, nor does it imply that Flask lacks capability. The “micro” in
microframework indicates that Flask intends to maintain a minimal, expandable
core.

APIs enable your product or service to communicate with other goods and
services without requiring you to understand their implementations. This may
facilitate app development simplification, therefore saving time and money.
When building new tools and products or managing current ones, APIs allow
flexibility, ease design, administration, and usage, and enable innovation.

Sometimes, APIs are seen like contracts, with documentation representing the
parties’ agreement: If party 1 delivers a remote request formatted in a
certain manner, party 2’s programme will answer. Because APIs make it easier
for developers to incorporate new application components into an existing
architecture, they facilitate collaboration between business and IT teams. In
response to everevolving digital marketplaces, where new rivals may transform
an entire sector with a single app, business requirements often undergo rapid
transformation. To remain competitive, it is essential to encourage the quick
development and implementation of new services. APIs are used to link a
microservices application architecture to cloud-native application
development, which is a determinable method for increasing development pace.

APIs are a streamlined method to link your own infrastructure through cloud-
native application development, but they also enable you to share your data
with clients and other external users. Public APIs provide distinctive
economic value since they may simplify and broaden how you communicate with
your partners, and possibly enable you to monetise your data (the Google Maps
API is a popular example).

One of the primary benefits of APIs is that they enable the separation of
functionality across systems. An API endpoint decouples the application using
a service from its underlying infrastructure. As long as the specification for
what the service provider is sending to the endpoint stays the same, the
changes to the infrastructure underlying the endpoint should go unnoticed by
the API-dependent apps. This provides the service provider with service
providing flexibility. For instance, if the API’s underlying architecture
consists of physical servers in a data centre, the service provider may simply
migrate to cloud-based virtual servers.

 **5.17 DeepFake Detection Methods incorporated on DeepSafe**

  1. MesoNet [15] is a CNN model that focuses on the mesoscopic characteristics of pictures. They provide two MesoNet variations, notably Meso and MesoInception. Meso employs typical convolutional layers, while Meso Inception is built on more complex Inception modules [3]; Meso Inception is included into the plat- form.
  2. FWA [16] is built on ResNet-50, which identifies DeepFake movies by revealing face warping distortions resulting from resizing and interpolation procedures.
  3. VA[18] addresses visual artefacts in face organs including the eyes, teeth, and facial features of synthetic faces. This approach is available in two variations: VAMLP and VA-LogReg. VA-MLP is based on a CNN whereas VA-LogReg em- ploys a logistic regression model. We include VA-MLP into the system.
  4. Xception is equipped with the FaceForensics++ dataset. It is a DeepFake de- tection approach based on the XceptionNet model. This approach offers the Xception-raw, Xception-c23, and Xception-c40 variations. Xception-raw is learned on raw videos, while Xception-c23 and Xception-c40 are trained on movies com- pressed to varying degrees. Xception-c23 is included in the platform.
  5. ClassNSeg [19] is a further CNN-based DeepFake detection approach that is de- signed to simultaneously identify forgeries pictures and segment modified re- gions.
  6. Capsule uses the VGG19 capsule structure as the primary architecture for Deep- Fake classification.
  7. DSP-FWA is a further enhanced approach based on FWA that includes a spatial pyramid pooling (SPP) module to better deal with the changes in face resolutions.
  8. CNNDetection [20] employs a conventional image classifier trained on just Pro- GAN [15], discovering that it generalises very well to unobserved topologies, datasets, and training techniques.
  9. According to Upconv [21], standard up-sampling approaches (upconvolution or transposed convolution) are incapable of accurately reproducing the spectral distributions of real training data. As a feature, they use the 2D amplitude spectrum and a simple SVM classifier.
  10. To generate predictions, WM combines two WS-DAN models (using EfficientNet- b3 and Xception feature extractors, respectively) and an Xception classifier.
  11. Selim employs a state-of-the-art encoder, EfficientNet B7, which has been pre trained with ImageNet and noisy student, and a heuristic method to pick 32 frames for each video to average predictions.

![](local/1*JFyb_HTUkSNZyShIF-vBfQ.png)

An illustration of how RestAPIs work in DeepSafe

![](local/1*oGTBvcuGiLDyHjUgNTjxOA.png)

The list of DeepFake detector incorporated in DeepSafe

# Discussion

One of the major aspects of developing DeepSafe was to address the ethical
concerns and create a system to detect fake media. The research question talks
about creating fake personas and putting systems and mechanisms in place to
comply with the fair usage of this technology. These systems need to be fused
with the fabric of social media and content sharing. Although, knowledge of
the existence of these technologies solves the major problem of seeing and
believing media content on the internet.

Media manipulation has been there since the start of media creation. Even
before deepfake it was possible to create such high quality videos, it was
just tedious and required professional experience. This technology just made
it easier to do so.

When Adobe Photoshop came into existence, media manipulation became common.
Adobe photoshop didn’t go extinct if anything it evolved to be more powerful
and feature rich than ever. Photoshop focuses more on images, DeepFakes
majorly on videos. The technological advancement will be analogous. Of course
creating reliable detectors are important but what is more important is being
aware that videos can be created and manipulated in high quality.

A few of our user journey interviews ended with feedback like including a
photoshop manipulation detector and audio deepfake detector onto the platform
as well. The future work could entail this.

The study focussed only on the platform and assumes that currently no
universal detector exists and the results may not be generalizable to sub
cases where a detector works for all the cases. URL and image matching can
computation but the most computation theoretically will be saved by video
matching.

DeepSafe works like a platform and can evolve over time without any major
changes. As long as an API can be written to run a model and save the
inference, DeepSafe will be relevant.

# Conclusion

DeepFake detection research is an extremely active domain. DeepSafe not only
helps keep track of the state of the art work until now but also provides a
forward facing development platform to add new ones built by researchers.

For end users, DeepSafe was tested in different versions to find out which one
is the most practical with the best user experience. We tested different
versions of chrome extension which would let users use it right in the
browser. The tests done with Androidand the ios app proved users want to use
it only when they feel suspicious of a video or photo. Running it in the
background the entire time ruins the user experience. The DeepSafe web app
which supports all the major systems including edge devices had the best user
experience. It had the lowest average inference time cause the processing was
server side and it supported the most devices hence could be used by most
number of users. Mobile app seems a great solution since it would mean using
the same device to consume digital content and for detection, hence lower
friction. The current mobile app has limited features hence we could not test
this hypothesis but something to explore in the future work.

DeepFake detection is an intermediate solution to a much bigger problem. Al-
though we need reliable detectors, what we need the most is educating the
general population about the existence and usage of this technology.

# Future Work

The platform incorporates eight cutting-edge detection algorithms, as well as
interfaces allowing researchers to implement their algorithms into the
platform. DeepSafe may continue to include more DeepFake detection algorithms
onto the platform in the future. Furthermore, multi-GPU systems may be used to
expedite the data analysis. APIs may be extended to provide more broad
detection algorithms for different media types. Audio-based deepfake detectors
may also be used since the detection principles are the same.

Blockchain systems store data in a decentralised, immutable ledger that is
regu- larly reviewed and re-confirmed by every organisation that uses it,
making it almost difficult to modify data once it has been established. One of
the most prominent uses of blockchain is to track the movement of
cryptocurrencies such as Bitcoin. However, since blockchain enables
decentralised authentication and a clear chain of custody, it has the
potential to be useful as a tool for monitoring and validating not just
financial resources, but also a wide range of other types of content.
Blockchain technology holds a lot of potential and can be explored in the
later version of DeepSafe. Blockchain has immensely benefited the fight
against deepfakes. However, it is not completely trust- worthy. To maximise
its potential, other technologies such as artificial intelligence will need to
be combined. Combining blockchain and artificial intelligence may dramatically
help the battle against deepfakes. Even if we consider blockchain’s
limitations in detecting deepfakes, it is still beneficial to have a system
that can aid in recognising deepfakes and thereby reducing the amount of
disinformation they distribute.

In the future versions of DeepSafe, sub video matching can be studied. We can
check for a video in the database to return it if it is real or fake without
running inferences.

