import tensorflow as tf
from keras.utils import Sequence
from keras.applications.resnet import preprocess_input
import pandas as pd
import os


class DataGenerator(Sequence):
    def __init__(self, df, batch_size, image_size):
        self.df = df
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths1 = self.df['p1'].tolist()
        self.image_paths2 = self.df['p2'].tolist()
        self.labels = self.df['labels'].values

    def __len__(self):
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        batch_image_paths1 = self.image_paths1[start_index:end_index]
        batch_image_paths2 = self.image_paths2[start_index:end_index]

        batch_images1 = self.load_and_preprocess_images(batch_image_paths1)
        batch_images2 = self.load_and_preprocess_images(batch_image_paths2)

        batch_labels = self.labels[start_index:end_index]

        return [batch_images1, batch_images2], batch_labels

    def load_and_preprocess_images(self, image_paths):
        batch_images = []

        for path in image_paths:
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, self.image_size)
            image = preprocess_input(image)
            batch_images.append(image)

        return tf.stack(batch_images)


def train_def():
    train_df = pd.read_csv('train-pairs.csv')

    p1_fold = []
    p2_fold = []
    labels = []

    for i, row in train_df.iterrows():
        p1_fold.append(row[0])
        p2_fold.append(row[1])

    p1_files = []
    p2_files = []

    for p1, p2 in zip(p1_fold, p2_fold):
        for i in os.listdir('train-faces/' + p1):
            for j in os.listdir('train-faces/' + p2):
                p1_files.append('train-faces/' + p1 + '/' + i)
                p2_files.append('train-faces/' + p2 + '/' + j)
                labels.append(1)

    df = pd.DataFrame({'p1': p1_files, 'p2': p2_files, 'labels': labels})
    df.to_csv("train_def.csv", index=False)
    print("Fine train")


def val_def():
    # Definisci il percorso del tuo set di immagini di validazione
    path_to_images = "val-faces"

    # Crea un DataFrame per salvare le etichette
    data = []
    columns = ["p1", "p2", "labels"]

    # Loop attraverso le cartelle famiglia
    for family_folder in sorted(os.listdir(path_to_images)):
        family_path = os.path.join(path_to_images, family_folder)
        if not os.path.isdir(family_path):
            continue

        # Ottieni la lista di sottocartelle per i membri della famiglia
        member_folders = sorted(os.listdir(family_path))

        # Loop attraverso le immagini di ogni membro della famiglia
        for i in range(len(member_folders)):
            member_folder1 = member_folders[i]
            member_path1 = os.path.join(family_path, member_folder1)
            if not os.path.isdir(member_path1):
                continue

            # Ottieni l'etichetta per il membro corrente (1 per parentela)
            label = 1

            # Loop attraverso gli altri membri della famiglia
            for j in range(i + 1, len(member_folders)):
                member_folder2 = member_folders[j]
                member_path2 = os.path.join(family_path, member_folder2)
                if not os.path.isdir(member_path2):
                    continue

                # Loop attraverso le immagini di ciascun membro
                for image_name1 in os.listdir(member_path1):
                    if not image_name1.endswith(".jpg"):
                        continue

                    for image_name2 in os.listdir(member_path2):
                        if not image_name2.endswith(".jpg"):
                            continue

                        # Aggiungi l'immagine1, immagine2 e l'etichetta al DataFrame
                        data.append([os.path.join(member_path1, image_name1), os.path.join(member_path2, image_name2),
                                     label])

    # Crea il DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Salva il DataFrame in un file CSV
    df.to_csv('val_def.csv', index=False)
