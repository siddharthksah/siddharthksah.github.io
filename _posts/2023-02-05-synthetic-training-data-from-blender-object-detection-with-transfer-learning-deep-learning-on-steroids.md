---
title: 'Synthetic training data from Blender + Object Detection with Transfer Learning = Deep Learning on Steroids!'
date: 2023-02-05
permalink: /posts/2023/02/synthetic-training-data-from-blender-object-detection-with-transfer-learning-deep-learning-on-steroids/
tags:
  - Synthetic Data
  - Blender
  - Generative AI
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
            Synthetic training data generated from Blender, combined with object detection using transfer learning, unleashes the potential of deep learning, amplifying its capabilities to unprecedented levels. This powerful synergy empowers AI models with an abundance of diverse and realistic data, fueling their accuracy and robustness for a wide array of applications.
        </p>
    </div>
    <div class="intro-image">
        <img src="https://github.com/siddharthksah/siddharthksah.github.io/blob/master/posts/synthetic-training-data-from-blender-object-detection-with-transfer-learning-deep-learning-on-steroids.jpg?raw=true">
        <p class="image-caption"><em>Image generated using text-to-image model by Adobe</em></p>
    </div>
</div>

A deep learning model is only as good as its data. If the data has inherent biases, the model will reflect them. The fine-tuning of models still has the original biases.

Collecting non-biased data is next to impossible; it is often woven into the very fabric of the data acquisition and generation.

Bias in training data is when a machine learning model treats different groups of people in different ways based on things like race, gender, or socioeconomic status. This happens when the training data used to build the model shows that bias and discrimination are common in society as a whole. Because of this, the model can keep and amplify these biases in the decisions and predictions it makes. For example, a biased training set for facial recognition could mean that people with darker skin tones will make more mistakes. It's important to deal with bias in training data to make sure that machine learning models are fair and reliable.

There are several ways to address biases in deep learning models:

1.  Diverse training data: Use a diverse and representative training dataset to ensure that the model is exposed to a range of examples from different groups and does not over-represent or under-represent any particular group.
2.  Fairness constraints: Apply mathematical constraints during the training process to ensure that the model makes fair predictions and decisions, such as equal opportunity or equal accuracy across different groups.
3.  De-biasing algorithms: Use algorithms that specifically target and remove bias from the model, such as adversarial training or re-sampling methods.
4.  Human review: Incorporate human review into the decision-making process to catch and correct any biases that may emerge.
5.  Monitoring and evaluation: Regularly monitor and evaluate the model’s performance and decision-making to identify and address any potential biases.

Apart from biases, deep learning models also have major data privacy concerns surrounding them.

Data privacy is a major concern with deep learning models because they often use large amounts of personal data to train and make predictions. This personal data can include sensitive information like financial records, medical records, or biometric data. If this information is not kept safe or is used in an unethical way, it can cause serious privacy violations and hurt people.

To address data privacy concerns in deep learning, several techniques can be used, such as:

1.  Data anonymisation: Remove or mask personal identifiable information from the data used for training.
2.  Data encryption: Encrypt sensitive data to protect it from unauthorised access.
3.  Data aggregation: Combine multiple data sources into a single, aggregate representation to reduce the risk of re-identification.
4.  Access control: Implement strict access controls to limit who can access the data and under what circumstances.
5.  Privacy preserving machine learning: Use privacy-preserving techniques such as differential privacy or federated learning to train models without compromising the privacy of individuals.

Synthetic data helps solve these problems. If implemented well, it can dampen both the biases and the data security concerns.

Let’s talk about adversarial attacks on deep learning models for a second. They are part of the data privacy concern. We can reverse-engineer our way to the data from the model it was trained on from only the model weights. Google came up with decentralised training called Federated Learning to solve these issues. In that case, sensitive data never leaves the original device.

There are several types of adversarial attacks, including:

*   Poisoning attacks: Injecting malicious data into the training set to cause the model to behave in a specific way when given similar inputs.
*   Evasion attacks: Modifying input data at test time in order to trick the model into making a wrong prediction.
*   Exploratory attacks: Attempting to find inputs that are misclassified by the model and understanding why this is the case.

Trained models can have backdoors too.

Alright, let’s talk some code now. In this article, we are going to create our own synthetic training data and use transfer learning to build an object detection model.

Let’s take an example. We have a factory that manufactures beverages. The factory recently came up with a new types of beverages. We want to classify them on the production floor so that we can pack them in different boxes. Now because we do not have any training data, we can wait to take photos for a few days and then sort them, but that is not feasible financially. How about we create the data? We have access to the 3D model of the bottles (Usually not a bottleneck). If we render them in different scenarios and use that as training data, it should work, right? Let’s see.

There are ways to generate synthetic data from Blender using Python. For this example, I am going to stick to open source datasets.

[https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset](https://www.kaggle.com/datasets/vencerlanz09/bottle-synthetic-images-dataset)

Let’s look at some synthetic data.

Class I — Beer Bottle
Class II — Plastic Bottle
Class III — Soda Bottle
Class IV — Water Bottle
Class V — Wine Bottle

A collage of five different classes. We have 5000 images per class. Some classes have 512X512 images and some 300X300 — we will handle that in the training process.

I will be using yolov5 to train my custom dataset and Weights & Biases to track the experiments.

The original dataset has the annotations in COCO JSON Format so we will need to convert it to YOLO TXT Format. I will be using [https://github.com/pylabel-project/samples](https://github.com/pylabel-project/samples) for this.

```python
!pip install pylabel
from pylabel import importer

# Copy images_raw to the working directory
# Note: This may take some time depending on the size of your images_raw folder
!cp -r ./input/bottle-synthetic-images-dataset/ImageClassesCombinedWithCOCOAnnotations/images_raw ./

# Copy annotations to the working directory
!cp -r ./input/bottle-synthetic-images-dataset/ImageClassesCombinedWithCOCOAnnotations/coco_instances.json ./

# Copy the test image to the output directory
!cp -r ./input/bottle-synthetic-images-dataset/ImageClassesCombinedWithCOCOAnnotations/test_image.jpg ./

# Specify the path to the coco.json file
path_to_annotations = r"./coco_instances.json"

# Specify the path to the images (if they are in a different folder than the annotations)
path_to_images = r"./images_raw"

# Import the dataset into the pylabel schema
dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="BCCD_coco")
dataset.df.head(5)

print(f"Number of images: {dataset.analyze.num_images}")
print(f"Number of classes: {dataset.analyze.num_classes}")
print(f"Classes: {dataset.analyze.classes}")
print(f"Class counts:\n{dataset.analyze.class_counts}")
print(f"Path to annotations:\n{dataset.path_to_annotations}")

try:
    display(dataset.visualize.ShowBoundingBoxes(2))
    display(dataset.visualize.ShowBoundingBoxes("./images_raw/00000002.jpg"))
except:
    pass

# This cell may take some time depending on the size of the dataset.
dataset.path_to_annotations = "labels"
dataset.export.ExportToYoloV5(output_path='text_files')

# Note!!! Only run this code once
# This will change the start of class numbers from 1 to 0
path = './text_files'  # Path of labels
labels = os.listdir(path)
for x in labels:
    lines = list()
    with open(path+"/"+x, "r+") as f:
        for line in f.read().splitlines():
            split_line = line.split(" ")  # Split on space character (and remove newline characters as well)
            split_line[0] = str(
                int(split_line[0]) - 1)  # Update the value inside the loop. The loop used in later is not needed.
            lines.append(split_line)  # Add split list into the list of lines

    with open(path+"/"+x, 'w') as file:  # Rewrite to file
        for line in lines:
            write_me = ' '.join(line)  # Use the join method to add the elements together
            file.write(write_me + "\n")
```

Now that we have the data ready in YOLO TXT format, We'll divide it into three parts: training, testing, and validation.

```python
# Read images and annotations
image_dir = r'./images_raw'
images = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
annotations = [os.path.join('./text_files', x) for x in os.listdir('./text_files') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the dataset into train-valid-test splits
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size=0.2, random_state=1)
val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size=0.5, random_state=1)

len(train_images), len(train_annotations)

# YOLOv5 expects annotations and images in directories in a specific format
# Let's first create those directories and then move the files into them
!mkdir images
!mkdir annotations
!mkdir images/train images/val images/test annotations/train annotations/val annotations/test

# Utility function to move images
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their folders
move_files_to_folder(train_images, 'images/train')
move_files_to_folder(val_images, 'images/val/')
move_files_to_folder(test_images, 'images/test/')
move_files_to_folder(train_annotations, 'annotations/train/')
move_files_to_folder(val_annotations, 'annotations/val/')
move_files_to_folder(test_annotations, 'annotations/test/')

!mv annotations labels
shutil.move("./images", "./yolov5")
shutil.move("./labels", "./yolov5")

# YOLOv5 has a dataset.yaml file which contains the directories of the training, test, and validation data
# Since we moved the files we will update the path in the yaml too

# Adjusting the parameters of the yaml file
yaml_params = {}
with open(r'dataset.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    yaml_file_list = yaml.load(file, Loader=yaml.FullLoader)
    yaml_params = yaml_file_list

yaml_params['path'] = 'images'
yaml_params['train'] = 'train'
yaml_params['val'] = 'val'
yaml_params['test'] = 'test'

# Overwriting the new params from the previous ones.
with open(r'dataset.yaml', 'w') as file:
    documents = yaml.dump(yaml_params, file)

# Moving the dataset.yaml inside the yolov5/data folder.
shutil.move("dataset.yaml", "yolov5/data")

shutil.move("./test_image.jpg", "./yolov5")

# Change the current directory inside the yolov5
%cd ./yolov5
```

Let’s talk about the training now. These are the hyperparameters we will tuning for this example.

1.  Size of the Image

2. Batch Size

3. Epochs

4. Workers

5. Model Architecture — There are four choices available: yolo5s.yaml, yolov5m.yaml, yolov5l.yaml, yolov5x.yaml.

```python
!python train.py --img 300 --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch 128 --epochs 100 --data dataset.yaml --weights yolov5s.pt --workers 24 --name yolo_bottle_det

# Running inference and testing if everything is ok.
!python detect.py --source images/test --weights runs/train/yolo_bottle_det/weights/best.pt --conf 0.25 --name yolo_bottle_det

detections_dir = "runs/detect/yolo_bottle_det/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15), subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    random_detection_image = PIL.Image.open(random.choice(detection_images))
    ax.imshow(random_detection_image)
```

Let’s make some predictions on real life data now.

```python
!python detect.py --source ./test_image.jpg --weights runs/train/yolo_bottle_det/weights/best.pt --conf 0.25 --name yolo_bottle_det

detections_dir = "runs/detect/yolo_bottle_det2/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]
random_detection_image = PIL.Image.open(random.choice(detection_images))
plt.figure(figsize=(30, 30))
plt.imshow(random_detection_image)
plt.xticks([])
plt.yticks([])
```
