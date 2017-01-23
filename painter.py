import argparse
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

#Parse Filename
parser = argparse.ArgumentParser()
parser.add_argument('filename',help='Name of the image')
args = parser.parse_args()

#Load the image into a matrix
name = args.filename
img = Image.open(name)
img_red,img_green,img_blue = img.split() #Splitting Channels of Image then converting to Numpy Array
img_red = np.array(img_red)
img_green = np.array(img_green)
img_blue = np.array(img_blue)
img_size = np.shape(img_red)  #Size of image as a tuple. Can choose blue or green channel as well
img_rolledoutsize = img_size[0] * img_size[1] #Size of rolled out vector

img_red = np.reshape(img_red,(img_rolledoutsize,1)) #Roll out the channel matrix into a vector
img_green = np.reshape(img_green,(img_rolledoutsize,1))
img_blue = np.reshape(img_blue,(img_rolledoutsize,1))

#Numpy Arrays of Input and output
y_values = np.concatenate((img_red,img_green,img_blue),axis=1)  #Array of true outputs
x_values = np.array([[j,i]for i in range(img_size[0]) for j in range(img_size[1])])

#Hyperparameters of the model
BATCH_SIZE = 1000 #Min-Batch size 
EPOCHS = 5000 #Number of epochs
INPUT_SIZE = 2 #Only x and y coordinates of pixel
HLAYER_SIZE = [20,20] #Simple 2 hidden layer architecture with 20 neurons each. Change the number of neurons here
OUTPUT_SIZE = 3 #RGB values of the pixel
TOTAL_STEPS = (img_rolledoutsize/BATCH_SIZE)*EPOCHS #Total number of steps to train

#The Model itself
x = tf.placeholder(tf.float32,[None,2]) #Tensor of inputs
y = tf.placeholder(tf.float32,[None,3]) #Tensor of true outputs

#W_i,b_i,a_i are weights bias and activations respectively of the ith layer. For more hidden layers, a list of tensors can be created before
W_1 = tf.Variable(tf.random_uniform([2,HLAYER_SIZE[0]],minval=0,maxval=1,dtype = tf.float32))
b_1 = tf.Variable(tf.random_uniform([HLAYER_SIZE[0]]))
a_1 = tf.nn.relu(tf.matmul(x,W_1)+b_1)

W_2 = tf.Variable(tf.random_uniform([HLAYER_SIZE[0],HLAYER_SIZE[1]],minval=0,maxval=1,dtype = tf.float32))
b_2 = tf.Variable(tf.random_uniform([HLAYER_SIZE[1]]))
a_2 = tf.nn.relu(tf.matmul(a_1,W_2)+b_2)

W_3 = tf.Variable(tf.random_uniform([HLAYER_SIZE[1],3],minval=0,maxval=1,dtype = tf.float32))
b_3 = tf.Variable(tf.random_uniform([3]))
y_hat = tf.nn.relu(tf.matmul(a_2,W_3)+b_3) #Tensor of predicted outputs

#L2 Loss for Regression
loss = tf.reduce_mean(tf.square(y_hat-y))

#Adam Optimization to minimize L2 Loss
train_step = tf.train.AdamOptimizer().minimize(loss)

#Training
present_epoch = 0 #Flag to keep track of epochs
final_painting = np.zeros((img_size[0],img_size[1],3)) #Array to store the final image as w x b x 3
with tf.Session() as sess:
	tf.global_variables_initializer().run()	
	for step in range(1,TOTAL_STEPS):
		present_epoch = int(((step-1)*BATCH_SIZE)/img_rolledoutsize)
		begin_point = ((step-1)*BATCH_SIZE)%img_rolledoutsize
		end_point = (step*BATCH_SIZE)%img_rolledoutsize
		x_batch = x_values[begin_point:end_point,:]
		y_batch = y_values[begin_point:end_point,:]
		sess.run(train_step, feed_dict={x: x_batch, y: y_batch}) #Feeding Data in Batches and performing a training step

		#Print Loss after every 50 steps		
		if(step%50 == 0):
			losses = (sess.run(loss, feed_dict={x: x_values, y: y_values})) 
			print 'Step: {} Loss: {}'.format(step,losses)
		
		#Output the present painting after every 100 epochs 
		if(present_epoch%100 == 0):
			painting_by_nn = sess.run(y_hat, feed_dict={x: x_values, y: y_values})
			#Formatting the output of ANN to that of a RGB image 
			final_painting[:,:,0] = np.reshape(painting_by_nn[:,0],(img_size[0],img_size[1]))
			final_painting[:,:,1] = np.reshape(painting_by_nn[:,1],(img_size[0],img_size[1]))
			final_painting[:,:,2] = np.reshape(painting_by_nn[:,2],(img_size[0],img_size[1]))
			result = Image.fromarray((final_painting).astype(np.uint8)) #Image in PIL library format
			result.save(name+'_painting_at_epoch'+str(present_epoch)+'.jpg')
