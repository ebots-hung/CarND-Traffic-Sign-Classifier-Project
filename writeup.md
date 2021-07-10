# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./1_data_visualization.png "Visualization"
[image2]: ./2_data_histogram.png "Data_Histogram"
[image3]: ./3_data_transl.png "Data_Translation"
[image4]: ./4_data_aumentation "Data_Aumentation"
[image5]: ./5_data_grayscale.png "Grayscale dataset"

[image6]: ./6_training_accuracy.png "Training Accuracy"

[image7]: ./7_my_images.png "My test Images"
[image8]: ./8_signtype_prediction.png "Sign Type Prediction"
[image9a]: ./9_top5_softmax_prob_0.png "Softmax Prob 0"
[image9b]: ./9_top5_softmax_prob_1.png "Softmax Prob 1"
[image9c]: ./9_top5_softmax_prob_2.png "Softmax Prob 2"
[image9d]: ./9_top5_softmax_prob_3.png "Softmax Prob 3"
[image9e]: ./9_top5_softmax_prob_4.png "Softmax Prob 4"
[image9f]: ./9_top5_softmax_prob_5.png "Softmax Prob 5"
[image10]: ./10_training_accuracy_comparison.png "Training Accuracy Comparison"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ebots-hung/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy methods to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. And a bar chart of data histogram. 

![Dataset Visualization][image1]

![Dataset Histogram][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I have experiment with 4 different datasets(no processing dataset, normalized dataset, grayscale normalized dataset and grayscale single channel dataset)

![Grayscale 3 channels dataset][image5]

![Data augmentation][image4]

With the normalized dataset, theoritically the data is standardized as a result training time and performance would be bettter. I can see that the training time for single channel grayscale dataset with dimension of (32x32x1) instead of (32x32x3), thus training time is much shortened, but still achieved the same performance result. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I started with LeNet 5 layers, then evolved to LeNetDropouts and LeNetSingleChannel.
My final model consisted of the following layers:

|Layer                  |         Description        |  Output  |
|-----------------------|----------------------------|----------|
|Input                  | RGB image                  | 32x32x3  |
|Convolutional Layer 1  | 1x1 strides, valid padding | 28x28x16 |
|RELU                   |                            |          |
|Max Pool               | 2x2                        | 14x14x16 |
|Convolutional Layer 2  | 1x1 strides, valid padding | 10x10x16 |
|RELU                   |                            |          |
|Max Pool               | 2x2                        | 5x5x16   |
|Fatten                 | To connect to fully-connected layers  |
|Fully-connected Layer 1|                            | 400      |
|RELU                   |                            |          | 
|Dropout                | 0.5 keep probability       |          |
|Fully-connected Layer 2|                            | 120      |
|RELU                   |                            |          |
|Dropout                | 0.5 keep probability       |          |
|Fully-connected Layer 3|                            | 43       |

Then upgrade to LeNet 7 layers model

| Layer         		|     Description	        					| Output  |
|:---------------------:|:---------------------------------------------:|----------|
| Input         		| 32x32x3 RGB image   							|           |
| Convolution 5x5     	| 1x1 stride, valid padding	                    | 28x28x6   | 
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding                  	| 14x14x10  |
| RELU					|												||
| Convolution 5x5     	| 1x1 stride, valid padding 	                | 8x8x16 |
| RELU					|												||
| Max Pooling       	| 2x2 stride, valid padding  	                | 4x4x16|
| Flatten				|	outputs 256									||
| Fully Connected     	| Input 256 and outputs 120                 	| 120 |
| RELU					|												||
| Dropout				|	keep_prob=0.5								||
| Fully Connected     	| Inputs 120 and outputs 100                 	| 100 |
| RELU					|												| 
| Fully Connected     	| Inputs 120 and outputs 100                 	| 100 |
| RELU					|												||
| Fully Connected     	| Inputs 84 and outputs 43                  	| 43|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an LeNet for the most part that was given, but I did add an additional convolution without a max pooling layer after it like in the udacity lesson.  I used the AdamOptimizer with a learning rate of 0.00095.  The epochs used was 27 while the batch size was 156. 

At first, I started with batchsize-128 and epochs number - 27, the training result was less then 0.9, then keep increasing batchsize to 156. The result improve much better, accuracy result reach 0.93. And I also tried to explore with LeNet 7 layers instead of 5, the performance also improved a bit to 0.943.
There was no much improvement when I changed the epoch number from 27 to 50, epoch 27 should be the best epoch number to save training time.  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Comparison between unnormalized vs normalizeddata:
* No pre-processing: 
** Train Accuracy = 0.998
** Validation Accuracy = 0.916
** Test Accuracy = 0.904
* No pre-processing (normalized dataset): 
** Train Accuracy = 0.943
** Validation Accuracy = 0.926
** Test Accuracy = 0.903

At first, I started with LeNet 5 layers and non preprocessed data. Then start tuning batchsize, learning rate. Then figure out that batchsize = 156 and rate = 0.0097 give the best result. 


Secondly, I tried with LeNet Dropout and pre-processed data by applying grayscale convertion/normalization. I also repeat the same with LeNet 7 layers and single channel grayscale dataset. 
 
Since the training activity consumes a lot of time as I execute the tensorflow on my personal computer.

Here is the comparison table between the methods I applied. (Eventhough I am not so happy with the accuracy result, I have to stop NN training after a week trial, hopefully I can improve some times later)

![NN training accuracy comparison][image10] 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![My test images][image7] 

The first image may be challenging, as it is a bit blur, the other images should be good enough to detect. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit - 30kmh   |	Speed limit - 30kmh							| 
| Bumpy road  			|   Bumpy road									|
| Ahead only     		|   Ahead only									|
| No vehicles      		|   No vehicles					 				|
| Go straight or left	|   Go straight or left							|
| General caution   	|   General caution  							|

The model was able to correctly guess 6 of the 6 traffic signs. However, the detection accuracy is only 0.833, much lesser than test set accuracy. The sign type 'No vehicles', detection probability is quite low, only 56%, because the number of trained images was lesser than other type. In order to improve the accuracy of new detection, I may need to improve training dataset by data augmentation or collect more data from the real scenarios.

![My prediction][image8] 


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .86         			| Speed limit - 30kmh   						| 
| .94     				| Bumpy road									|
| .98					| Ahead only									|
| .56	      			| No vehicles					 				|
| .96				    | Go straight or left     						|
| .100				    | General caution      							|

![Probability_0][image9a] 
![Probability_1][image9b] 
![Probability_2][image9c] 
![Probability_3][image9d] 
![Probability_4][image9e] 
![Probability_5][image9f] 

My traffic sign 1 - Speed limit 30kmh, detection probability is 86%.
My traffic sign 2 + 3 + 5 + 6 - detection probability ( 94%, 98%, 96% and 100%)is quite good, much better than the test set, because of the quality of the traffic images and details features have the good matches with training set. 
My traffice sign 4 - No vehicles, detection probability is pretty low, 56%. 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


