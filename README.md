# ISYE-6740-Project---Deep Neural Network for Shoe Model Identification 

**Project Team Member Names:** Vivi Banh & Chukwuemeka Okoli 

 
**Project Title:** Deep Neural Network for Shoe Model Identification 


**Project Goal:** The goal of this project is to build a deep neural network that can identify shoe models from images. The network will be trained on a dataset of images of shoes, and it will be able to classify new images with high accuracy. 

 
**Problem Statement**

Our project aims to develop a system that can accurately identify Nike shoe models worn by models in photos and provide consumers with the name of the shoe models. This will enable users to easily find and purchase the specific shoe models they are interested in, enhancing their shopping experience. The shoe models we will be focusing on identifying are the Air Jordan 1, Air Force 1, and Air Max 1. 

 

**Project Scope**

The scope of the project will involve the following tasks: 

 - Collecting dataset of images of shoes.  
 - Preprocessing the images to prepare them for training. 
 - Designing and training a deep neural network using the Convolutional Neural Network (CNN) architecture. 
 - Testing the performance of the CNN model. 
 - Discussing the deployment of the trained model to a production environment. 

**Data Source**

The data for this project is the UT Zappos50K Shoe Dataset (ver 1.2) created by Yu and Grauman (2014) and hosted on Kaggle.com. The UT Zappos50K is a large shoe dataset consisting of 50,025 catalog images collected from Zappos.com. The dataset provides a collection of images featuring different shoe models including the famous Nike brand. We will be using images specifically of Air Jordan 1, Air Force 1, and Air Max 1 for designing and training the model. Additionally, a separate set of images featuring models wearing various Nike shoes and non-Nike shoes will be used to test the model's performance. 

  

**Methodology** 

We will utilize a Convolutional Neural Network (CNN) model, a deep learning architecture well-suited for image classification tasks. The CNN model will be trained using the labeled shoe images to learn the distinctive features of each shoe model. We will create our model using Python 3.10. The process can be summarized as follows: 

 + Data preprocessing: The shoe images will undergo preprocessing steps, including resizing and normalization, to ensure they are in a suitable format for training. Data augmentation techniques such as vertical and horizontal reflections, rotation up to 90 degrees, and vertical and horizontal shifting of the images up to 20% of their original size will be applied to enhance the model's ability to generalize. 

 + Model architecture: A CNN model will be designed, consisting of convolutional layers, pooling layers, and fully connected layers. The architecture will be tailored to capture the distinctive visual features of Nike shoe models. Techniques such as dropout and batch normalization may be incorporated to improve the model's generalization and prevent overfitting. 

 + Model training: The designed CNN model will be trained using the labeled shoe images as input and their corresponding shoe model names as target labels. The model will learn to recognize the visual patterns associated with each shoe model during the training process. Training will utilize optimization algorithms such as stochastic gradient descent (SGD). 

 + Model evaluation: The trained CNN model will be evaluated using a separate set of test images containing models wearing various Nike shoes. The accuracy of the model in correctly identifying the shoe models will be measured using evaluation metrics such as categorical cross-entropy loss, accuracy, precision, recall, and F1 score. 

 + Fine-tuning: If necessary, the model will be fine-tuned by adjusting hyperparameters such as learning rate, batch size, or the number of layers to improve its accuracy and optimize its performance. 

 + Deployment: Considerations for deploying the trained model into a production environment (ex: potential integration with a user-facing application) will be discussed if time permits. 

  

**Evaluation and Final Results**

The accuracy of the model will be evaluated by comparing its predictions for identifying shoe models with the true labels of the test images using evaluation metrics such as categorical cross-entropy, accuracy, precision, recall, and F1 score. 

The final result will consist of a trained model capable of accurately identifying Nike shoe models worn by models in photos. The success of the project will be determined by the model's ability to achieve a high level of accuracy in identifying the correct shoe models. The model can be deployed as a user-facing application, allowing consumers to upload images and receive the corresponding shoe model names as output. 

**References** 

Albawi, S., Mohammed, T. A., & Al-Zawi, S. (2017). Understanding of a Convolutional Neural Network. 2017 International Conference on Engineering and Technology (ICET), 1–6. https://doi.org/10.1109/icengtechnol.2017.8308186  

Yu, A., & Grauman, K. (2014). Fine-Grained Visual Comparisons with Local Learning. Computer Vision and Pattern Recognition. https://doi.org/10.1109/cvpr.2014.32 


‌ 


