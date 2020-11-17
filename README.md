# face-attribute-prediction
Face Attribute Prediction on [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) benchmark with PyTorch Implemantation.
## Dependencies

* Anaconda3 (Python 3.6+, with Numpy etc.)
* PyTorch 0.4+
* tensorboard, tensorboardX

## Dataset

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset is a large-scale face dataset with attribute-based annotations. Cropped and aligned face regions are utilized as the training source. 


## Features
* Good capacity as well as generalization ability.
* Achieve 92%+ average accuracy on CelebA Val.
* Both ResNet and EfficientNet as the backbone for scalability
* fast convergence: 91% acc on CelebA Val after 1 epoch.
* To-do: Focal Loss
* To-Do: Class balanced sampler
* To-do: BCE loss for attributes recognition.
