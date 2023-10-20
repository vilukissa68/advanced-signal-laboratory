# Image Classification

## Requirements

In the root of the projct:
``` shell
pip install -r requirements.txt
```

## Dataset
To be able to train the network we need to have the GENKI-4K dataset.
In the root of the project

``` shell
sh scripts/get_dataset.sh
```
This downloads the wanted dataset and untars it to the correct directory. Before the data is ready to be used the dataset needs to be serialized.
Inthe root of the project

``` shell
cd src
python main.py --serialize
```
This will take a while. After the process had been finished the model is ready to be trained.

## Training
To train the models as described in the assignment instructions, run this is in /src directory:

``` shell
python main.py --train
```

After training is done the trained model is added into the /models directory in the root of the project. Commandline arguments can be used to change training parameters. For example this command runs the based model for 150 epochs with a learning rate of 0.0002 with batches of 64 images: 

``` shell
python main.py --train --epochs=150 --lr=0.0002 --batch_size=64 
```

All the options can be shown with the command:

``` shell
python main.py --help
```

### GPU acceleration
The system supports GPU acceleration with the --device argument to speed up training.

If you have a Nvidia GPU that support cuda:
``` shell
python main.py --train --device=cuda
```

For M1/M2 macbooks one can use the MPS backend with:
``` shell
python main.py --train --device=mps
```

## Prediction
After the model has been trained the prediction system can be ran using the following command in the /src directory

``` shell
python main.py --load_model="{name of the model}.model"
```

The predictor can be existed by pressing q
