# Classification and Detection with Convolutional Neural Networks

This project is aiming to use different Convolutional Neural Network based methods: CNN, VGG16 with pretrained weights, and VGG16 from scratch, to detect digits in given street house images. And the images/digits used for test purpose in this project has different scale, fonts, positions, orientation, light conditions, noise conditions. The model was trained through Tensorflow GPU on university's scluster and only need to run one function python code for the detection process.

### How to run

build conda environment:

`conda env create -f cv_proj.yml`

`conda activate cv_proj`

run for five images detection:

`python run.py`

run for video detection:

`python video.py`

### Files

* [Training Model (TensorFlow) - blue step](https://github.com/chd415/Computer-Vision/blob/master/Classification-and-Detection-with-Convolutional-Neural-Networks/Qiao-tf.py)

* [Training Model (PyTorch)  - blue step](https://github.com/chd415/Computer-Vision/blob/master/Classification-and-Detection-with-Convolutional-Neural-Networks/Qiao-pytorch.py)

* [Function code (Python) - green step](https://github.com/chd415/Computer-Version/blob/master/Classification-and-Detection-with-Convolutional-Neural-Networks/run.py)

* [Project report](https://github.com/chd415/Computer-Version/blob/master/Classification-and-Detection-with-Convolutional-Neural-Networks/report.pdf)


### Results Example

* [Digit detection on video](https://youtu.be/TdwY1eImXxg)

* Model architecture
<img src="./licence/flowchart.png" width="850" />

* Images with correct/incorrect detection

<img src="./licence/fiveimage.png" width="850" />
