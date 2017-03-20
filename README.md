AlexNet_tf
==========
A implement of AlexNet(A typical Convolution Neural Network) with TensorFlow v1.0. It's used to extract features from images for CBIR Task/Object Retrieve on VGG Paris Dataset.

Features
--------
* Read raw images and build auto-pipeline(Queue) producing batch for input, no OpenCV API, pure TF OPs. The official website no reading example like this.
* Support TensorBoard for Visualization. It's convenient to observe loss curve and dataflow graph of the model.
* Using tf.app.flags to parse cmd arguments.
* A integrate data processing: Data Input => Model Build => Model Training => Neural Network Feedforward(Features Extraction) => Object Retrieve

Reference
---------
1. AlexNet by Alex Krizhevsky: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
2. Features extraction idea: http://dl.acm.org/citation.cfm?id=2654948&CFID=913983328&CFTOKEN=27220803
3. TensorFlow whitepaper: http://download.tensorflow.org/paper/whitepaper2015.pdf
4. TensorFlow Cifar10 example: https://www.tensorflow.org/versions/master/tutorials/deep_cnn/
5. VGG Paris Dataset: http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/
6. Good documents to understand Convolution:
 * https://deeplearning4j.org/convolutionalnets
 * https://cs231n.github.io/convolutional-networks/

Contact
-------
My Email: hiocde@gmail.com
Or post issues if you find problems.