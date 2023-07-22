
> medium-to-markdown@0.0.3 convert
> node index.js https://siddharthksah.medium.com/synthetic-training-data-object-detection-with-transfer-learning-deep-learning-on-steroids-e20f76bd4269

Synthetic training data from Blender + Object Detection with Transfer Learning = Deep Learning on Steroids!
===========================================================================================================

[![Siddharth Sah](https://miro.medium.com/v2/resize:fill:88:88/1*RTWoIcWgxb9qaY9qBDHikA.jpeg)

](https://medium.com/?source=post_page-----e20f76bd4269--------------------------------)

[Siddharth Sah](https://medium.com/?source=post_page-----e20f76bd4269--------------------------------)

·

[Follow](https://medium.com/m/signin?actionUrl=https%3A%2F%2Fmedium.com%2F_%2Fsubscribe%2Fuser%2F1aa18b2cd060&operation=register&redirect=https%3A%2F%2Fsiddharthksah.medium.com%2Fsynthetic-training-data-object-detection-with-transfer-learning-deep-learning-on-steroids-e20f76bd4269&user=Siddharth+Sah&userId=1aa18b2cd060&source=post_page-1aa18b2cd060----e20f76bd4269---------------------post_header-----------)

8 min read·Feb 5

\--

Listen

Share

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

Class I — Beer BottleClass II — Plastic BottleType III — Soda BottleType IV — Water BottleType V — Wine Bottle

A collage of five different classes. We have 5000 images per class. Some classes have 512X512 images and some 300X300 — we will handle that in the training process.

I will be using yolov5 to train my custom dataset and Weights & Biases to track the experiments.

The original dataset has the annotations in COCO JSON Format so we will need to convert it to YOLO TXT Format. I will be using [https://github.com/pylabel-project/samples](https://github.com/pylabel-project/samples) for this.

```
!pip install pylabel  
from pylabel import importer  
  
\# Copy images\_raw to working directory  
\# Note: This may take some time depending on the size of your images\_raw folder  
!cp -r ./input/bottle-synthetic-images-dataset/ImageClassesCombinedWithCOCOAnnotations/images\_raw ./  
  
\# Copy annotations to working directory  
!cp -r ./input/bottle-synthetic-images-dataset/ImageClassesCombinedWithCOCOAnnotations/coco\_instances.json ./  
  
\# Copy test image to output directory  
!cp -r ./input/bottle-synthetic-images-dataset/ImageClassesCombinedWithCOCOAnnotations/test\_image.jpg ./  
  
#Specify path to the coco.json file  
path\_to\_annotations = r"./coco\_instances.json"  
#Specify the path to the images (if they are in a different folder than the annotations)  
path\_to\_images = r"./images\_raw"  
  
#Import the dataset into the pylable schema   
dataset = importer.ImportCoco(path\_to\_annotations, path\_to\_images=path\_to\_images, name="BCCD\_coco")  
dataset.df.head(5)  
  
print(f"Number of images: {dataset.analyze.num\_images}")  
print(f"Number of classes: {dataset.analyze.num\_classes}")  
print(f"Classes:{dataset.analyze.classes}")  
print(f"Class counts:\\n{dataset.analyze.class\_counts}")  
print(f"Path to annotations:\\n{dataset.path\_to\_annotations}")  
  
try:  
    display(dataset.visualize.ShowBoundingBoxes(2))  
    display(dataset.visualize.ShowBoundingBoxes("./images\_raw/00000002.jpg"))  
except:  
    pass  
  
\# This cell may take some time depending on the size of the dataset.  
dataset.path\_to\_annotations = "labels"  
dataset.export.ExportToYoloV5(output\_path='text\_files');  
  
\# Note!!! Only run this code once  
\# this will change the start of class numbers from 1 to 0  
path = './text\_files' #path of labels  
labels = os.listdir(path)  
for x in labels:  
    lines = list()  
    with open(path+"/"+x, "r+") as f:  
        for line in f.read().splitlines():  
            split\_line = line.split(" ")  # split on space character (and remove newline characters as well)  
            split\_line\[0\] = str(  
              int(split\_line\[0\]) - 1)  # update the value inside the loop. the loop used in later not needed.  
            lines.append(split\_line)  # add split list into list of lines  
  
    with open(path+"/"+x, 'w') as file:  # rewrite to file  
        for line in lines:  
            write\_me = ' '.join(line)  # Use join method to add the element together  
            file.write(write\_me + "\\n")
```

Now that we have the data ready in YOLO TXT format, We'll divide it into three parts: training, testing, and validation.

```
\# Read images and annotations  
image\_dir = r'./images\_raw'  
images = \[os.path.join(image\_dir, x) for x in os.listdir(image\_dir)\]  
annotations = \[os.path.join('./text\_files', x) for x in os.listdir('./text\_files') if x\[-3:\] == "txt"\]  
  
images.sort()  
annotations.sort()  
  
\# Split the dataset into train-valid-test splits   
train\_images, val\_images, train\_annotations, val\_annotations = train\_test\_split(images, annotations, test\_size = 0.2, random\_state = 1)  
val\_images, test\_images, val\_annotations, test\_annotations = train\_test\_split(val\_images, val\_annotations, test\_size = 0.5, random\_state = 1)  
  
len(train\_images),len(train\_annotations)  
  
\# yolov5 expects annotations and images in directories in a specific format  
\# let's first create those directories and then move the files into them  
!mkdir images  
!mkdir annotations  
!mkdir images/train images/val images/test annotations/train annotations/val annotations/test  
  
#Utility function to move images   
def move\_files\_to\_folder(list\_of\_files, destination\_folder):  
    for f in list\_of\_files:  
        try:  
            shutil.move(f, destination\_folder)  
        except:  
            print(f)  
            assert False  
  
\# Move the splits into their folders  
move\_files\_to\_folder(train\_images, 'images/train')  
move\_files\_to\_folder(val\_images, 'images/val/')  
move\_files\_to\_folder(test\_images, 'images/test/')  
move\_files\_to\_folder(train\_annotations, 'annotations/train/')  
move\_files\_to\_folder(val\_annotations, 'annotations/val/')  
move\_files\_to\_folder(test\_annotations, 'annotations/test/')  
  
!mv annotations labels  
shutil.move("./images", "./yolov5")  
shutil.move("./labels", "./yolov5")  
  
\# Yolov5 has a dataset.yaml file which containes the directories of the training, test and validation data  
\# Since we moved the files we will update the path in the yaml too  
  
\# Viewing the original unprocessed yaml file  
\# yaml\_params = {}  
\# with open(r'dataset.yaml') as file:  
\#    # The FullLoader parameter handles the conversion from YAML  
\#    # scalar values to Python the dictionary format  
\#    yaml\_file\_list = yaml.load(file, Loader=yaml.FullLoader)  
\#    yaml\_params = yaml\_file\_list  
\#    print(yaml\_file\_list)  
  
\# Adjusting the parameters of the yaml file  
yaml\_params\['path'\] = 'images'  
yaml\_params\['train'\] = 'train'  
yaml\_params\['val'\] = 'val'  
yaml\_params\['test'\] = 'test'  
yaml\_params  
  
\# Overwriting the new params from the previous ones.  
with open(r'dataset.yaml', 'w') as file:  
    documents = yaml.dump(yaml\_params, file)  
  
\# Moving the dataset.yaml inside the yolov5/data folder.  
shutil.move("dataset.yaml", "yolov5/data")  
  
shutil.move("./test\_image.jpg", "./yolov5")  
  
\# Change the current directory inisde the yolov5  
%cd ./yolov5
```

Let’s talk about the training now. These are the hyperparameters we will tuning for this example.

1.  Size of the Image

2\. Batch Size

3\. Epochs

4\. Workers

5\. Model Architecture — There are four choices available: yolo5s.yaml, yolov5m.yaml, yolov5l.yaml, yolov5x.yaml.

```
!python train.py --img 300 --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml --batch 128 --epochs 100 --data dataset.yaml --weights yolov5s.pt --workers 24 --name yolo\_bottle\_det
```

Running inference and testing if everything is ok.

```
!python detect.py --source images/test --weights runs/train/yolo\_bottle\_det/weights/best.pt --conf 0.25 --name yolo\_bottle\_det  
  
detections\_dir = "runs/detect/yolo\_bottle\_det/"  
detection\_images = \[os.path.join(detections\_dir, x) for x in os.listdir(detections\_dir)\]  
  
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15),  
                        subplot\_kw={'xticks': \[\], 'yticks': \[\]})  
  
for i, ax in enumerate(axes.flat):  
    random\_detection\_image = PIL.Image.open(random.choice(detection\_images))  
    ax.imshow(random\_detection\_image)
```

Let’s make some predictions on real life data now.

```
!python detect.py --source ./test\_image.jpg --weights runs/train/yolo\_bottle\_det/weights/best.pt --conf 0.25 --name yolo\_bottle\_det  
  
detections\_dir = "runs/detect/yolo\_bottle\_det2/"  
detection\_images = \[os.path.join(detections\_dir, x) for x in os.listdir(detections\_dir)\]  
random\_detection\_image = PIL.Image.open(random.choice(detection\_images))  
plt.figure(figsize=(30,30));  
plt.imshow(random\_detection\_image)  
plt.xticks(\[\])  
plt.yticks(\[\]);
```
