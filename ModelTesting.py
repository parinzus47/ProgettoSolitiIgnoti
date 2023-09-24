from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import itertools
import gc
from keras.applications.resnet import preprocess_input
from ModelTraining import prewhiten_facenet


def test_couples():
    model = load_model('TrainedModel/best_model.h5')

    # Imposta il percorso dei due file di immagine da confrontare
    img1_path = 'test-faces/face3118.jpg'
    img2_path = 'test-faces/face51.jpg'

    # Carica le due immagini e convertile in un formato adatto per l'input del modello
    img1 = cv2.imread(img1_path)
    img1 = cv2.resize(img1, (160, 160))
    #img1 = prewhiten_facenet(img1)
    img1 = np.expand_dims(img1, axis=0)

    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(img2, (160, 160))
    #img2 = prewhiten_facenet(img2)
    img2 = np.expand_dims(img2, axis=0)

    # Usa il modello per prevedere la parentela tra le due immagini
    prediction = model.predict([img1, img2])

    # Stampa la previsione
    print(prediction)


def accuracy_test():
    model = load_model('TrainedModel/best_modelFaceNetPrewhiten.h5')

    test_df = pd.read_excel('Dataset/test_competition.xlsx')

    correct_predictions = 0
    total_predictions = 0

    for index, row in itertools.islice(test_df.iterrows(), 10000):
        if row[3] != "gmgs" and row[3] != "gmgd" and row[3] != "gfgs" and row[3] != "gfgd":

            img1_path = 'test-faces/' + row[1]
            img2_path = 'test-faces/' + row[2]

            # Carica le due immagini e convertile in un formato adatto per l'input del modello
            img1 = cv2.imread(img1_path)
            img1 = cv2.resize(img1, (160, 160))
            #img1 = tf.io.read_file(img1_path)
            #img1 = tf.image.decode_jpeg(img1, channels=3)
            #img1 = tf.image.resize(img1, (160, 160))
            #img1 = preprocess_input(img1)
            img1 = prewhiten_facenet(img1)
            img1 = np.expand_dims(img1, axis=0)

            img2 = cv2.imread(img2_path)
            img2 = cv2.resize(img2, (160, 160))
            #img2 = tf.io.read_file(img2_path)
            #img2 = tf.image.decode_jpeg(img2, channels=3)
            #img2 = tf.image.resize(img2, (160, 160))
            #img2 = preprocess_input(img2)
            img2 = prewhiten_facenet(img2)
            img2 = np.expand_dims(img2, axis=0)

            # Usa il modello per prevedere la parentela tra le due immagini
            prediction = model.predict([img1, img2])
            print(prediction)

            if prediction > 0.5:
                correct_predictions += 1

            total_predictions += 1
            gc.collect()

    accuracy = correct_predictions / total_predictions
    print('Accuracy:', accuracy)
    print("Su casi totali: ", total_predictions)


def parente_mist():
    model = load_model('TrainedModel/best_modelInceptionResNetV2.h5')
    print(model.layers[1].input)
    puntata = "05-04-23"

    partecipanti = []
    for i in range(1, 9):
        partecipante = 'C:/Users/simom/Desktop/Immagini parenti misteriosi/' + puntata + '/Par' + str(i) + '.png'
        partecipanti.append(partecipante)

    #parente_misterioso = tf.io.read_file('C:/Users/simom/Desktop/Immagini parenti misteriosi/' + puntata + '/ParM.png')
    #parente_misterioso = tf.image.decode_jpeg(parente_misterioso, channels=3)
    #parente_misterioso = tf.image.resize(parente_misterioso, (224, 224))
    parente_misterioso = cv2.imread('C:/Users/simom/Desktop/Immagini parenti misteriosi/' + puntata + '/ParM.png')
    parente_misterioso = cv2.resize(parente_misterioso, (160, 160))
    #parente_misterioso = prewhiten_facenet(parente_misterioso)
    parente_misterioso = preprocess_input(parente_misterioso)
    parente_misterioso = np.expand_dims(parente_misterioso, axis=0)

    val_predizioni = []

    for partecipante in partecipanti:
        #immagine_partecipante = tf.io.read_file(partecipante)
        #immagine_partecipante = tf.image.decode_jpeg(immagine_partecipante, channels=3)
        #immagine_partecipante = tf.image.resize(immagine_partecipante, (160, 160))
        #immagine_partecipante = numpy.expand_dims(immagine_partecipante, axis=0)
        immagine_partecipante = cv2.imread(partecipante)
        immagine_partecipante = cv2.resize(immagine_partecipante, (150, 150))
        #immagine_partecipante = prewhiten_facenet(immagine_partecipante)
        immagine_partecipante = preprocess_input(immagine_partecipante)
        immagine_partecipante = np.expand_dims(immagine_partecipante, axis=0)

        partecipante_rappresentazione = model.predict([immagine_partecipante, parente_misterioso])
        val_predizioni.append(partecipante_rappresentazione)

        # Stampa la predizione
        print(f"Predizione per {partecipante}: {partecipante_rappresentazione}")

    print("Il parente misterioso è il n: " + str(np.argmax(val_predizioni) + 1))
    soluzione = open('C:/Users/simom/Desktop/Immagini parenti misteriosi/' + puntata + '/sol.txt', "r").read()
    print("La soluzione vera è: " + soluzione)
