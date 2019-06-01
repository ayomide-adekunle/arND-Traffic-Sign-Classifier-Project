# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/eighty.png "Traffic Sign 1"
[image5]: ./test/priority_road.png "Traffic Sign 2"
[image6]: ./test/right_of_way.png "Traffic Sign 3"
[image7]: ./test/road_work.png "Traffic Sign 4"
[image8]: ./test/stop.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how Class Distribution for the Training Data.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color information of an image is not needed to classify the image. Converting to grayscale will make the process easier and faster.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it is needed to reduce the size of numbers I used during my implementation.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:
a) The augumented data is converted to grayscale
b) The argumented data is normalized



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following parameters:

EPOCHS = 50
BATCH_SIZE = 128
learning_rate = 0.001
sigma = 0.1

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.





I used the LeNet architecture for the project. This architecture is relevant to the traffic sign application because it shows good result when tested.
To get a good result, I had to chage the EPOCHS many times until I got a good result for EPOCHS = 50
The accuracy of the model when tested with five different road signs was 100%

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image was difficult to classify for some reasons. The model sees it as 50 speed limit while it is 80. I had to adjust some network variables and retrain my model to classify it accurately.

The second image which is "priority road" might be difficult because to classify because it looks like the yield sign

The third image which is "rigth of way at the next intersection" has some background in it that might confuse the model. This image also has a shape close to other signs.

The fourth image which is "road work" also have a background and shape of the signs.

The fifth image which is " Stop sign" is clearly visible but has text information.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority Road     		| Priority Road   									| 
| Right of way    			| Right of way										|
| Road Work				| Road Work											|
| Stop Sign	      		| Stop Sign					 				|
| 80 speed limit		| 80 speed limit     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority Road sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        			    | Priority Road  								| 
| .44     				| Yield										    |
| .035					| Round about mandatory							|
| .008	      			| No entry					 				    |
| .0005				    | Stop      							        |

For the second image, the model is relatively sure that this is a Right of way sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        			            | Right of way 							| 
| 7.1474486e-29    				| Beware of ice/snow					|
| 0					            | Speed limit(20km/hr)					|
| 0	      			            | Speed limit(30km/hr)					|
| 0				                | Speed limit(50km/hr)      			|

For the third image, the model is relatively sure that this is a Road work sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1        			            | Road work 							| 
| 7.0150549e-16   				| Speed limit(70km/hr)					|
| 2.3770005e-18					| Speed limit(80km/hr)					|
| 1.5304534e-22      			| No passing for vehicles over 3.5 metric tons |
| 1.3614264e-22				    | Wild animal crossing      			|


For the fourth image, the model is relatively sure that this is a Stop sign (probability of 0.76). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 7.6798087e-01       			| Stop 							        | 
| 2.3197551e-01   				| Turn left ahead					    |
| 3.5563855e-05				    | No passing					        |
| 7.4138084e-06      			| Ahead only                            |
| 7.2874303e-07				    | Yield     			                |

For the fifth image, the model is relatively sure that this is a Speed limit(80km/hr)  sign (probability of 1). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1      			    | Speed limit(80km/hr)  						| 
| 9.779572e-16  		| Speed limit(50km/hr) 					        |
| 6.756073e-31			| Speed limit(50km/hr) 					        |
| 0      			    | Speed limit(20km/hr)                          |
| 0			            | Speed limit(60km/hr)    			            |



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


