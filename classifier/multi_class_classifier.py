import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3


class MultiClassClassifier(object):
    def __init__(self, dir):
        self.dir = dir
        self.data_dir = self.dir + "/dcm_data"
        self.database_dir = self.dir + "/dcm_database"
        self.train_dir = self.database_dir + "/train"
        self.mapped_dir = self.database_dir + "/train_mapped"
        self.model_dir = self.database_dir + "/model/multi_class_classifier"
        self.checkpoint_dir = self.model_dir + "/checkpoint/multi_class_classifier"
        self.tflite_dir = self.model_dir + "/tflite"

    def loadData(self, is_mapped=False):
        src_dir = self.mapped_dir if is_mapped else self.train_dir
        if not os.path.isdir(src_dir):
            print("{} doesn't exist".format(src_dir))
            if is_mapped:
                print(
                    "Consider running tools/mapping.py to create train dataset mapping.")
            else:
                print("Consider running dcm to generate train dataset")
            exit()
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            src_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE)
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            src_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE)
        self.class_names = self.train_ds.class_names

    def loadModel(self, title):
        saved_dir = os.path.join(self.model_dir, title)
        self.model = tf.keras.models.load_model(saved_dir)
        with open(os.path.join(saved_dir, 'class_names.names'), 'rb') as fp:
            self.class_names = pickle.load(fp)

    def loadModelQuantized(self, title):
        model_path = os.path.join(self.tflite_dir, title+'_f16.tflite')
        with open(os.path.join(self.tflite_dir, title+'.names'), 'rb') as fp:
            self.class_names = pickle.load(fp)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def loadWeights(self, title):
        checkpoint = title + ".cpkt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
        self.model.load_weights(checkpoint_path).expect_partial()

    def createModel(self, learning_rate=0.0005):
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)

        num_classes = len(self.class_names)
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomTranslation(
                (-0.1, 0.1), (-0.1, 0.1), fill_mode='constant'),
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ])

        self.base_model = tf.keras.applications.resnet_v2.ResNet101V2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL),
                                                                        include_top=False,
                                                                        weights='imagenet')
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        self.base_model.trainable = False

        inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
        x = self.data_augmentation(inputs)
        x = preprocess_input(x)
        x = self.base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes)(x)
        self.model = tf.keras.Model(inputs, outputs)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
            metrics=['accuracy'])
        self.model.summary()

    def finetuneModel(self, learning_rate=0.00005):
        self.base_model.trainable = True
        fine_tune_at = 300
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
            metrics=['accuracy'])
        self.model.summary()

    def trainModel(self, epochs, title, prev_history=None):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            initial_epoch=0 if prev_history == None else prev_history.epoch[-1],
        )
        return history

    def evaluateModel(self):
        loss, acc = self.model.evaluate(self.val_ds)
        print("Model accuracy: {:5.2f}%".format(100 * acc))
        return loss, acc

    def quantizeModel(self, title):
        os.makedirs(self.tflite_dir, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        with open(os.path.join(self.tflite_dir, title+'_f16.tflite'), 'wb') as f:
            f.write(tflite_model)
        with open(os.path.join(self.tflite_dir, title+'.names'), 'wb') as fp:
            pickle.dump(self.class_names, fp)

    def predictSingle(self, img):
        img = (np.expand_dims(img, 0))
        probability_model = tf.keras.Sequential([self.model,
                                                 tf.keras.layers.Softmax()])
        predictions = probability_model(img, training=False).numpy()
        return predictions[0], self.class_names

    def predictSingleQuantized(self, img):
        img = (np.expand_dims(img, 0)).astype(np.float32)
        input_index = self.interpreter.get_input_details()[0]["index"]
        output_index = self.interpreter.get_output_details()[0]["index"]
        self.interpreter.set_tensor(input_index, img)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(output_index)
        logits = tf.convert_to_tensor(logits, dtype=tf.float32)
        predictions = tf.nn.softmax(logits).numpy()
        return predictions[0], self.class_names

    def visualizeData(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(25):
                ax = plt.subplot(5, 5, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.show()

    def visualizeAugmented(self):
        plt.figure(figsize=(10, 10))
        for images, labels in self.train_ds.take(1):
            for i in range(25):
                augmented_images = self.data_augmentation(images)
                ax = plt.subplot(5, 5, i + 1)
                plt.imshow(augmented_images[i].numpy().astype("uint8"))
                plt.title(self.class_names[labels[i]])
                plt.axis("off")
        plt.show()

    def visualizeTraining(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def runTraining(self, title='resnet', epochs=100, is_mapped=False):
        initial_epochs = int(epochs/2)
        learning_rate = 0.0005
        title = "mapped_" + title if is_mapped else title

        self.loadData(is_mapped)
        self.createModel( learning_rate=learning_rate)
        history = self.trainModel(
            initial_epochs, title=title)
        self.visualizeTraining(history)
        self.evaluateModel()
        self.finetuneModel(learning_rate=learning_rate/10)
        history_fine = self.trainModel(
            epochs, title=title+'_finetuned', prev_history=history)
        self.visualizeTraining(history_fine)
        self.evaluateModel()
        self.quantizeModel(title=title+'_finetuned')

    def runFineTuning(self, title='resnet', epochs=50):
        learning_rate = 0.00005
        self.loadData()
        self.createModel()
        self.loadWeights(title=title)
        self.finetuneModel( learning_rate=learning_rate)
        history = self.trainModel(
            epochs, title=title+'_finetuned')
        self.visualizeTraining(history)
        self.evaluateModel()
        self.quantizeModel(title=title+'_finetuned')

    def runHpTuning(self, title='resnet', epochs=10, finetune=False):
        self.loadData()
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005]
        losses = []
        accuracies = []
        histories = []
        for lr in learning_rates:
            self.createModel( learning_rate=lr)
            if finetune:
                self.loadWeights(title=title)
                self.finetuneModel( learning_rate=lr)
            history = self.model.fit(
                self.train_ds,
                validation_data=self.val_ds,
                epochs=epochs)
            loss, acc = self.evaluateModel()
            losses.append(loss)
            accuracies.append(acc)
            histories.append(history)
        for i in range(len(losses)):
            print("Lr:", learning_rates[i], "Loss:",
                  losses[i], "Acc:", accuracies[i])
            self.visualizeTraining(histories[i])
