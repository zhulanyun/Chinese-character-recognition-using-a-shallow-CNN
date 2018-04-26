# Chinese-character-recognition-using-a-shallow-CNN
  This is shallow convolution neural network used to identify Chinese characters. The Chinese characters used for training and testing not only contain modern Chinese characters, but also contain different kinds of Chinese characters in various historical periods, such as Oracle, official script, etc. Train sets and test sets can be downloaded from https://pan.baidu.com/s/1iA-7JCBff1ZE8PGUpVks6w and the password is sewi. 
  This model is a very shallow convolution neural network, can be easily trained on the CPU, especially for students who have just learned the deep learning or Keras framework as a primer. The correct rate for this model on top5 is 94.12, which is efficient in neural networks of similar depths.
  Before training this model, you need to convert the original dataset into a 28*28 image format and integrate all the images into the npy file form.


