# Tensorflow - Image Classification Wrapper

This is a simple image classifier API to train and test image classification models. 

Supported models:
- AlexNet 
- VGG16
- Inception (In progress)
- SqueezeNet (In progress)

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

## Utilities

To download images:
```
	python utils/google_download.py --outpath path/to/output/data --keywords add,your,keywords --prefixes prefix --limit 1000 -chrome /path/to/chromedriver
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