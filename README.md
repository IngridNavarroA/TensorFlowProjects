# TensorFlow Image Classifier API 

This is a simple image classifier API to train and test image classifier models. 

Supported models:
- AlexNet 
- VGG16
- Inception (In progress)
- ResNet (In progress)
- SqueezeNet (In progress)

You can also implement your own model using the layer wrappers, and add more wrappers as you need them. 

The API allows to perform end-to-end training, finetuning and restoring a training process. I am implementing a GUI to test images on trained models. In the meantime, use the test.py if you want to test a given model.

## Requirements and installation

This code has been implemented / tested / maintained with:
- OpenCV == 3.4
- Python3 
- Tensorflow == 1.13.1

Installing a virtual environment using python3 using bash:
```
	sudo pip3 install virtualenv virtualenvwrapper
	echo "# Virtual Environment Wrapper"  >> ~/.bashrc
	echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
	source ~/.bashrc
```
Or using zsh:
```
	sudo pip3 install virtualenv virtualenvwrapper
	echo "# Virtual Environment Wrapper"  >> ~/.zshrc
	echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.zshrc
	source ~/.zshrc
```

Create virtual environment with python3:
```
	mkvirtualenv tf-classifier -p python3
	workon tf-classifier
```

Test python version on virtual enviornment:
```
	workon tf-classifier
	python -V
	deactivate
```

Also, I provided a list of requirements, which you can install as follows: 
```
	workon tf-classifier
	pip install -r requirements.txt
```

Note: OpenCV is not included in these requirements. I follow this [tutorial](https://www.learnopencv.com/install-opencv3-on-ubuntu/) to install it, except I used version 3.4.

## Utils

I designed a couple of utilities to fetch and prepare the dataset for training. 

### Data Fetch 

This is to fetch a dataset from the server where I stored the images. 

```
	python utils/fetch_dataset_parallel.py --outpath data/

```

Or

```
	python utils/fetch_dataset_serial.py --outpath data/
```

# ---------------------------------------------
# ADD TO README: Classification framework and Training process. 

### Data Processing 

These are scripts to change dataset colorspaces and perform offline data augmentation. 

```
	python utils/preprocess.py --inpath path/to/input/data --outpath path/to/output/data --crop v
```

### Data Split

This code is to split data into train and test sets. Furthermore, the train set is divided into train and validation during the training process. 

```
	python utils/split.py --inpath path/to/input/data --opath path/to/output/data --test_set 0.2
```


### Entrenamiento de la red (end-to-end)
```
	workon tf-classifier
  python src/classifier/train.py --data path/to/training/data --max_iter 20000 --gpu 0 --restore 0 --fintune 0
```

### Finetunning de la red
```
	workon tf-classifier
	python src/train.py --data path/to/training/data --max_iter 20000 --gpu 0 --restore 0 --fintune 1
```

### Restaurar entrenamiento de la red
```
	workon tf-classifier
	python src/train.py --data path/to/training/data --max_iter 20000 --gpu 0 --restore 1
```

### Visualización de entrenamiento 
```
	workon tf-classifier
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


### TODOS:
-[x] Implementar Alexnet 
-[x] Implementar VGG16 con entrenamiento end-to-end, restore y finetuning 
-[ ] Implementar ResNet con entrenamiento end-to-end, restore y finetuning 
-[ ] Implementar Inception con entrenamiento end-to-end, restore y finetuning 
-[ ] Implementar SqueezeNet con entrenamiento end-to-end, restore y finetuning 
-[ ] Conseguir modelo pre-entreando para Inception, ResNet y SqueezeNet
-[ ] Implementar finetuning para Inception
-[ ] Documentacion de codigo - wiki
-[ ] Agregar bash scripts para correr pruebas las redes
-[ ] Agregar funcionalidad de entrenamiento multiescala 
-[ ] Implement GUI to test models
-[ ] Modular data fetch