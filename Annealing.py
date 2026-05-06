import tensorflow as tf
import numpy as np
import re
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.spatial.distance import cosine
import os
import argparse
from models import ResNetN
from models import ResNetN2
import keras
from utils import custom_functions as func
from utils import custom_callbacks as cb
import math

def fix_scheduler(epoch, lr):
    if 100 <= epoch < 150:  
        lr = 0.001  
    elif epoch >= 150:  
        lr = 0.0001  

    return lr

if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    seed = 12227
    func.set_seeds(seed)  # Set seeds for repeatability

    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='ResNet56')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch')
    parser.add_argument('--epoch', type=int, default=1)

    args = parser.parse_args()
    architecture = args.architecture
    dataset_name = args.dataset
    weights_dir = args.weights if args.weights != '' else f'./weights/{dataset_name}/{architecture}'
    model_name = args.model_name
    verbose = args.verbose
    epoch_annealing = args.epoch

    model_name = architecture.split('/')[-1] if model_name == '' else model_name
    print(f"{model_name} {dataset_name}", flush=True)

    # Create directory for saving weights if it doesn't exist
    os.makedirs(weights_dir, exist_ok=True)

    if dataset_name == 'CIFAR100': 
        cifar = 100
    elif dataset_name == 'CIFAR10':
        cifar = 10

    # Load the CIFAR-10/100 dataset
    X_train, y_train, X_test, y_test = func.cifar_resnet_data(cifar=cifar)

    # Determinar num_classes baseado no dataset
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    num_classes = len(y_train[0])
    print(num_classes)

    # Determinar N_layers baseado na arquitetura
    match = re.search(r'ResNet(\d+)', architecture)
    if match:
        N_layers = int(match.group(1))
        print(f"\nNumber of Layers: {N_layers}", flush=True)
    else:
        raise ValueError(
            "Arquitetura não suportada. Suporta apenas formatos como ResNetXX, onde XX é o número de camadas.")

    # Load a model
    if N_layers == 18:
        inputs = keras.Input(shape=(32, 32, 3))
        outputs = ResNetN2.resnet18(inputs)
        model = keras.Model(inputs, outputs)
    else:
        model = ResNetN.build_model(model_name, input_shape=(32, 32, 3), num_classes=num_classes, N_layers=N_layers)

    # Path to random starting weights
    # random_weights_path = os.path.join(weights_dir, f'@random_starting_weights_{model_name}_.weights.h5')
    
    # Load or create the random starting weights using the seed
    # func.load_or_create_weights(model, random_weights_path)

    weights_path = os.path.join(weights_dir, f'{model_name}_{dataset_name}_epoch_{epoch_annealing}.weights.h5')
    print(f"Loading weights from {weights_path}.")
    model.load_weights(weights_path)

    # Configure the optimizer
    lr = 0.01
    sgd = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True) #, decay=1e-6
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    # Configure Data Augmentation
    datagen = func.generate_data_augmentation(X_train)

    # Set up learning rate scheduler
    lr_scheduler_callback = LearningRateScheduler(fix_scheduler)

    # Set up the ModelCheckpoint callback to save the model every n epochs
    # checkpoint_callback = ModelCheckpoint(
    #     filepath=os.path.join(weights_dir, f"{model_name}_{dataset_name}_epoch_{{epoch:02d}}.weights.h5"),
    #     save_weights_only=False,  # Save only the weights
    #     save_freq='epoch',  # Save at the end of every epoch
    #     verbose=1
    # )
    
    callbacks = [lr_scheduler_callback]     # , checkpoint_callback

    # Set manual epoch loop configuration
    epochs = 200
    batch_size = 16
    epoch_annealing = epoch_annealing + 1

    # Repeat the data k times, datagen will transform
    k = 3
    y_aug = np.tile(y_train, (k, 1))
    X_aug = np.tile(X_train, (k, 1, 1, 1))
    
    # Epoch loop
    for epoch in range(epoch_annealing, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        model.fit(
            datagen.flow(X_aug, y_aug, batch_size=batch_size, seed=seed, shuffle=True),
            epochs=epoch, initial_epoch=epoch - 1,
            verbose=verbose, callbacks=callbacks,
            validation_data = (X_test, y_test),
            validation_freq=5
        )

    # Evaluate model after training
    y_pred = model.predict(X_test, verbose=0)
    accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    print(f'Final accuracy: {accuracy:.4f}')
