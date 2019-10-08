# Tensorflow - Image Classification Wrapper

This is a simple image classifier API to train and test image classification models. 

Supported models:
- AlexNet 
- VGG16
- Inception (In progress)

You can also implement your own model using the layer and module wrappers, and add more wrappers as you need them. 

The API allows to perform end-to-end training, fine-tuning and restoring a training process. To test the model use the test.py

## Requirements and installation

This code has been implemented / tested / maintained with:
- OpenCV >= 3.4
- Python3.6
- Tensorflow == 1.14.0

Installing a virtual environment using python3 using zsh:
```
	sudo pip3 install virtualenv virtualenvwrapper
	echo "# Virtual Environment Wrapper"  >> ~/.zshrc
	echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.zshrc
	source ~/.zshrc
```

Create virtual environment with python3:
```
	mkvirtualenv classifier -p python3
```

Test python version on virtual enviornment:
```
	workon classifier
	python -V
	deactivate
```

Also, I provided a list of requirements, which you can install as follows: 
```
	workon classifier
	pip install -r requirements.txt
```

Note: OpenCV is not included in these requirements. I follow this [tutorial](https://www.learnopencv.com/install-opencv3-on-ubuntu/) to install it, except I used version 3.4.

## Tools

In the tools folder, I provided a set of scripts data preprocessing. Such scripts 
allow to;
 - Download images based on Google search queries
 - Remove repeated files
 - Rename files
 - Crop images
 - Delete images 
 - Split a dataset into train set and test set
 - Perform off-line data augmentation

Below, I show usage examples:

#### Download data 
This script is used to perform an automatic dataset download using the 
[google-images-download package](https://github.com/hardikvasa/google-images-download ). 
This script allows to download multiple datasets at a time, by providing a comma separated
set of keywords. To use it, run:

To use it, run:
```
	python tools/google_download.py --outpath path/to/output/data --keywords kw1,kw2,... --prefixes superhero --limit 20000 --chrome /usr/lib/chromium-browser/chromedriver
```

#### Pre-process data
The script pre_process_data.py allows to review a dataset from a given folder. It
allows to select a region of interest on the image and only keep that region. Also, 
it allows to delete unnecessary images. To use it run:
```
	python tools/pre_process_data.py --inpath path/to/input/data
```
The script will show each of the images on the input folder and will allow to 
perform the following operations on each image:

<p align="center"><img src="./readme/preprocess_ops.png" /> </p>

NOTE: When cropping an image it will look like:

<p align="center"><img src="./readme/crop_sample.png" /> </p>

#### Removed repeated 
This script is useful when using the google-image-download multiple times for 
similar queries. The package might download the same image several times but with 
different indexes. The script will remove the repeated images. To use it, run:
```
	python tools/remove_repeated.py --inpath path/to/input/data
```

#### Rename 
This script is used to rename all files in an input directory. It uses a basename
provided by the user and adds current date and hour to the filename to avoid
overwriting. To use it, run:
```
	python tools/rename.py --inpath data/superhero/flash --name flash
```

### Entrenamiento de la red (end-to-end)
```
	workon classifier
  python classifier/train.py --data path/to/training/data --max_iter 20000 --gpu 0
```

### Finetunning de la red
```
	workon classifier
	python clasifier/train.py --data path/to/training/data --max_iter 20000 --gpu 0 --fintune
```

### Restaurar entrenamiento de la red
```
	workon classifier
	python clasifier/train.py --data path/to/training/data --max_iter 20000 --gpu 0 --restore
```

### Visualización de entrenamiento 
```
	workon classifier
	tensorboard --logdir=log
```

### Prueba de modelo entrenado
```
	workon tf-classifier
	python src/test.py --data path/to/testing/data --meta path/to/model
```

Al abrir en la dirección de loopback (e.g. http://127.0.1.1:6006) indicada por TensorBoard, se observarán gráficas como las siguientes:

### Accuracy
<p align="center"><img src="./readme/accuracy.png" /> </p>

### Loss
<p align="center"><img src="./readme/loss.png" /> </p>

## Arquitectura actual del clasificador
<p align="center"><img src="./readme/alexnet.png" /> </p>