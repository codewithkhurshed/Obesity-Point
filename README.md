# Obesity-Point
We present the design and evaluation of culturally <b> Obesity Risk Prediction based on Human Activity Recognition</b> , designed as a tool to help people predict their possible risk of obesity from our solution. 
In this thesis, we present the design and evaluation of culturally Obesity Risk Prediction
based on Human Activity Recognition, designed as a tool to help people predict their
possible risk of obesity from our solution. Users can set plans for seven days, 15 days, 30
days, and more to track their daily physical activity to predict a timeline for obesity. To
create our proposed deep learning model, we have developed a deep learning technique
based on three layered LSTM in google colab.

## Developer Info:
1. Diganta Chowdhury. CSE, EDU. Email: digantachowdhury074@gmail.com.
2. Khorshed Alam. CSE, EDU. Email: mohdkhurshed120@gmail.com.
3. Promila Hoque. Lecturer, EDU. Email: promila@eastdelta.edu.bd


## 1. Starting in figure with Splash Screen and Login Page. If the user is already registered, they will be able to Login.
<img src="https://user-images.githubusercontent.com/115551112/216057048-754e29e9-0e02-4118-bd19-d93fba7e68fa.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216057231-700c2fc6-0459-4b97-99e5-274536694389.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216057307-d8b87be3-5f3f-4d21-a497-2aa3b228cbaa.png" width="200" height="400">
<br>
<br>
## 2. In the Registration Process in figure, we collect the primary user's info and food consumption per day calorie from the user

<img src="https://user-images.githubusercontent.com/115551112/216059256-414f63c5-091e-45a6-8ed2-ab4f6a632ae2.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059192-6021ce1c-a283-4dfb-892f-1099af0e06ca.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059224-6a185d2f-970b-4469-ad60-ca9d9a91e2c9.png" width="200" height="400">
<br>
<br>
## 3. For better help, we have provided a calorie food chart of Bangladeshi food items. After Login successfully, the user can access our application's dashboard in figure. 

<img src="https://user-images.githubusercontent.com/115551112/216059285-266851be-6e84-4d4b-b021-f142bdaf401d.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059316-f0404d05-93e4-46c8-9ca4-b360ecfe48b0.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059381-3dac6d9c-13e1-4ea5-bc67-9914ff8c9c51.png" width="200" height="400">
<br>
<br>
## 4. User can start a plan for tracking their physical activity to track their obesity timeline.

<img src="https://user-images.githubusercontent.com/115551112/216059438-18e68cd2-ac27-4713-b4bc-84f88afd6ef5.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059477-780fd043-cb05-445c-b4b4-19eaccc435dc.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059516-c556893d-dfda-4902-a84b-62b90a2eade3.png" width="200" height="400">
<br>
<br>
## 5. After Finishing the Plan, it will calculate the inputs and show the result of signs of obesity up to 1 month, six months, and one year.

<img src="https://user-images.githubusercontent.com/115551112/216059557-07d6354a-a0c7-4616-b76c-4bebdd3e1a65.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059607-c2a78dfe-e534-44d6-a3e7-8c5a174d23e6.png" width="200" height="400"> <img src="https://user-images.githubusercontent.com/115551112/216059649-c0f19964-5e37-4d0a-a057-f5a6387c7f8d.png" width="200" height="400">

# Proposed Solution
![Proposed Solution Diagram1 (XML) (1)](https://user-images.githubusercontent.com/115551112/216069785-2e39b283-71f5-4f57-bc9a-55533ef2c740.jpg)



The methodology of this paper is briefly described in this chapter of the report. There must be an algorithm to track Human Activity Recognition with six classes. The classes are sitting, standing, walking, jogging, upstairs, and downstairs. Data has been collected by an accelerometer sensor in the left pocket and right pocket. We have used three layered LSTM (Long Short-Term Memory) to predict the possible outcome of human Activity among six classes. The workflow of our proposed solution is divided into three parts: 
(1) Dataset Formation. 
(2) Training Phase. 
(3) Result Phase.
## Dataset Formation:
![image](https://user-images.githubusercontent.com/115551112/216064558-1becad7f-e5ac-45e3-8536-bbe6eb33ddad.png)

In this research, we have used only one dataset called WISDM Lab Activity Sensor Dataset shown figure . There is the total of 1098207 entries. The number of classes is 6. The classes are sitting, standing, walking, jogging, upstairs, and downstairs. For each class, there are 424,400 entries for walking, 342,177 for jogging, 122,869 for upstairs, 100,427 for downstairs, 59,939 for sitting, and 48,395 for standing. The number of missing attributes is none.
![image](https://user-images.githubusercontent.com/115551112/216064858-212803dd-8ead-422b-a418-00fe4447b542.png)
## Data Preprocessing
![image](https://user-images.githubusercontent.com/115551112/216064957-e2c98775-a314-435e-8b12-f333983fc94f.png)
To get the maximum out of this dataset, we have performed preprocessing to our data in figure 3.6. We have sorted the data according to user and timestamps. Then, we divided our dataset into sub-datasets. Each dataset carries 200 points. The total sub-dataset is 54,901. The number of train datasets is 43920, and Test Dataset is 10981.

## Training Phase:
A Long Short-Term Memory network is a type of recurrent neural network that can hold over long-term input data sequences. LSTM is good for those problems where we rely on long input data sequences. So, let's examine how LSTM operate from figure:
![image](https://user-images.githubusercontent.com/115551112/216065561-d5e854f4-b79f-4545-968b-c8f863a9d68f.png)

c_t-1  is the previous cell state where c_t is the next cell state. h_t-1  is the previous hidden state and h_t is the next hidden state. x_t are our current inputs. In our case it is accelerometer data and timestamps. Forget gate is [8] a type of state where cell decides which information must be keep or forget by taking the weights and biases with respect to x_t in sigmoid activation function. So that it can return values from 0 to 1. The closer to zero means forget and the closer to 1 means keep. To determine c_t , we need the output of input states and cross product of the output of forget gate which is  c_ti  & f_t.  After adding is  c_ti  & f_t  we will get the next cell state is  c_t. For Input states, we take inputs of x_t   and previous hidden state values h_t-1  to determine i_t  & g_t. For input gate i_t   we use sigmoid activation function and for input node g_t  we use tanh activation function. For output state, we take inputs of x_t   and previous hidden state values h_t-1  to determine o_t. An output from current cell state c_t passed through tanh activation function with cross product to the output of o_t is the next hidden state h_t. Consider the situation when we need to update some information in a calendar. An RNN applies a function to the existing data to accomplish this, entirely transforming it. While LSTM just performs minor adjustments to the data through cell state-based addition or multiplication. This is how LSTM selectively forgets and recalls information, beating RNNs.
 In our case, 200-time steps are a good fit. Then we defined our model as a sequential Keras model and added 3 LSTM hidden layers seen in figure 3.8. The final output layer will give us the predicted class result. We have used Adam optimizer for training our deep learning model. It is widely used for stochastic gradient descent for training deep learning models. [26]. Due to a lack of resources, we could only use 50 epochs, which was more than enough for our purposes once our model fits an n number of epochs. In our case, n = 50 and batch size = 4096 with a number of hidden units = 128.
 
 # HAR Model Deployment in Android
 We have successfully created our model in google colab. Now, we have exported our model in (.pb) format file and deployed it in Android Studio for our Android application to check our model on the real-time accelerometer sensor data. It showed an outstanding outcome in the left pocket and right pocket data.
 
# Flowchart
 ![image](https://user-images.githubusercontent.com/115551112/216065971-0cf6aca1-aece-4c5d-8be9-0a6fa28f98d4.png)
# ER Diagram
![image](https://user-images.githubusercontent.com/115551112/216066032-4cceb80a-0a8f-4dc7-8658-c294aa8603c3.png)

# Result Analysis of HAR Model
Our Human Activity Recognition has achieved 99.20% validation accuracy for our LSTM-based deep learning model. Here we can see the validation accuracy details, and classification matrix for each activity predicted by our model.
![image](https://user-images.githubusercontent.com/115551112/216066367-0bd9f3dc-5aa3-4f72-93fd-fa0ed9499cc4.png)
![image](https://user-images.githubusercontent.com/115551112/216066404-d439b4fd-0c00-4795-b021-7abff73f2d9e.png)
![image](https://user-images.githubusercontent.com/115551112/216066437-6d5191b4-aa61-4281-adb9-247fb16d2705.png)
![image](https://user-images.githubusercontent.com/115551112/216066481-6f75483d-136c-4e37-9caa-265ebf4984f0.png)


                                          







