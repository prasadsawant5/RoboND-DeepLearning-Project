# RoboND-Follow-Me-Project

## Network Architecture
The attempt was to build a model which is as lean as possible. I'm emphasizing on keeping the model lean so that it can be deployed on mobile devices like a cell phone, drone (although very ambitious) or on an edge device like Jetson TX2. Keeping this in mind, the FCN consists of three encodes, one 1x1 convolution and then followed by three decoders. 

### Skip Connections
Skip Connections are what give the FCN the ability to retain information from previous layers. With Skip Connections we can use the output from an encoder layer as the input for a decoder layer. This allows the model to learn from multiple resolutions which helps it retain details that could have been lost. Thus, increasing our accuracy on our segmentation.

## Model

The input dta has a resolution of 160x160. The first encoder layer halves the resolution and uses 24 filters overall. Thus, the output of first encoder is 80x80x24. The second encoder does the same thing and outputs a resolution of 40x40x48. Again, the final encoder does the exact same thing and we get a resolution of 20x20x96. The 1x1 convolution doesn't impact the resolution of data. This data of dimension 20x20x96 is fed to the first decoder layer which gives an output dimension of 40x40x96. The second and third decoder does the exact same thing giving a dimension of 80x80x48 and 160x160x24 respectively. Finally, the output layer gives a resolution of 160x160x3. 

![Model](/images/FCN_model.jpg)

## Network Parameters

I started with the standard value of learning rate, i.e. 0.001. However, the loss ratio between training loss and validation loss was relatively high. A value between 0.003 and 0.004 gave an optimal ratio.  I started the batch_size at 32 and settled on 128 as being big enough for a stable amount of data. I felt that the smaller batch sizes would be less consistent in training and increase the validation loss. The model converged with 100 epoches. Thus, didn't felt the need to increase it. I set the steps_per_epoch to be the number of images / batch size. I increased the number of workers to 8 just to speed up the learning time.

Therefore, my current parameters are,

1. learning_rate = 0.004  
2. batch_size = 128  
3. num_epochs = 100  
4. steps_per_epoch = 32  
5. validation_steps = 50  
6. workers = 8

![Loss](/images/loss.png)

## Training

The model was trained locally on my rig which has an NVIDIA GPU. The model is trained to classify and segment humans only. If we use images of animals or vehicles it would not be able to detect them accurately. We would have to retrain the model to look for these objects. We would also want to change the model to be able to tell us what we are identifying.

## Future Enhancements
As we know that neural networks are data hungry, feeding it more training data can increase accuracy. Also, quality of data matters as well. Each image in the training data needs to be as much distinctive as possible. The reason being, a diverse dataset can help the network learn some useful signals from the images. However, we need to realize that adding more training data would result in changing the architecture and the parameters. 

We can also use a different architecture such as faster R-CNN to train such a model.



