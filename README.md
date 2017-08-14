# Synopsis

This repository contains Deep Learning code to understand DL concepts like CNN, R-CNN, GANs etc. 

# General Architecture
On the highest levels ('cnn_model.py') there is a general Neural Network architecture. Within this you can use the specific variables, weights, and the layers belonging to the LeNet5, AlexNet of VGGNet-16 architectures. 
![general model](/figures/tensorflow_model.png?raw=true "Tensorfow model architecture")

## LeNet5 / LeNet5-like CNN:
![LeNet5 architecture](/figures/lenet5_architecture.png?raw=true "LeNet5 Architecture")

The LeNet5 architectures have been trained with the MNIST, CIFAR-10 and Oxflower-17 datasets and yield the following accuracies:
![LeNet5 test score on MNIST](/figures/mnist_test_score.png?raw=true)
![LeNet5 test score on CIFAR-10](/figures/cifar10_test_score.png?raw=true)
![LeNet5 test score on CIFAR-10 for different optimizers](/figures/cifar10_test_score_optimizers.png?raw=true)

![LeNet5 test score on CIFAR-10 for RMS optimizer](/figures/cifar10_test_accuracies_rms_optimizer.png?raw=true)
![LeNet5 test score on CIFAR-10 for Gradient Descent Optimizer](/figures/cifar10_test_accuracies_gd_optimizer.png?raw=true)
![LeNet5 test score on CIFAR-10 for Adam optimizer](/figures/cifar10_test_accuracies_adam_optimizer.png?raw=true)
![LeNet5 test score on CIFAR-10 for Adagrad optimizer](/figures/cifar10_test_accuracies_adag_optimizer.png?raw=true)



## AlexNet
![AlexNet architecture](/figures/alexnet_architecture.png?raw=true "AlexNet Architecture")

The AlexNet architecture haS been trained with the Oxflower-17 dataset for different optimizers:
![AlexNet test score on Oxflower-17 dataset for different optimizers](/figures/alexnet_test_score_different_optimizers.png?raw=true)



## VGGNet-16
![VGGNet-16 architecture](/figures/vggnet_architecture.png?raw=true "VGGNet-16 Architecture")
