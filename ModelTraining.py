import pandas as pd
import numpy as np
import random
from DataGenerator import DataGenerator
from vggface import VGGFace
from keras import Model, Input
from keras.layers import Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, Subtract, Multiply, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def randomizer(csv_file, case):
    # Leggi il file CSV
    df = pd.read_csv(csv_file)

    # Esegui il campionamento casuale con ripristino separatamente per etichetta 0 e 1
    label_0_rows = df[df['labels'] == 0].sample(n=int(case/2), random_state=random.seed())
    label_1_rows = df[df['labels'] == 1].sample(n=int(case/2), random_state=random.seed())

    # Combina le righe con etichetta 0 e 1
    final_df = pd.concat([label_0_rows, label_1_rows]).reset_index(drop=True)

    return final_df


def prewhiten_facenet(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def model_train():
    n_dataset = 529080
    train_df = randomizer("train_def.csv", n_dataset)
    validation_df = randomizer("val_def.csv", (n_dataset * 0.2))
    train_gen = DataGenerator(train_df, 32, (224, 224))
    validation_dt = DataGenerator(validation_df, 32, (224, 224))

    #base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    #base_model = load_model("C:/Users/simom/Downloads/epoch17.h5")
    #base_model = Model(inputs=base_model.input, outputs=base_model.layers[-4].output)
    #base_model = InceptionResNetV1(input_shape=(160, 160, 3), weights_path="C:/Users/simom/Downloads/facenet_keras_weights.h5")
    #base_model = Model(inputs=base_model.input, outputs=base_model.layers[-3].output)
    base_model = VGGFace(architecture="resnet50", include_top=False, input_shape=(224, 224, 3)).model
    base_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    for layer in base_model.layers[:-2]:
        layer.trainable = False

    input_1 = Input(shape=(224, 224, 3))
    input_2 = Input(shape=(224, 224, 3))

    x1 = base_model(input_1)
    x2 = base_model(input_2)

    x1 = Concatenate(axis=-1)([GlobalAveragePooling2D()(x1), GlobalMaxPooling2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalAveragePooling2D()(x2), GlobalMaxPooling2D()(x2)])

    #x1 = Concatenate(axis=-1)([x1, x1])
    #x1 = GlobalAveragePooling2D()(x1)
    #x2 = Concatenate(axis=-1)([x2, x2])
    #x2 = GlobalAveragePooling2D()(x2)

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x = Multiply()([x1, x2])

    x = Concatenate(axis=-1)([x, x3])

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.01)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model([input_1, input_2], output)

    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=Adam(0.00001))
    model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001)

    model.fit(train_gen, batch_size=32, epochs=100, validation_data=validation_dt,
              callbacks=[early_stopping, model_checkpoint, reduce_lr])