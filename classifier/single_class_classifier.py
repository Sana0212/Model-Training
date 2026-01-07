import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import shutil
import time
from sklearn.metrics import classification_report, confusion_matrix, f1_score

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 128
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3


class SingleClassClassifier(object):
    def __init__(self, dir):
        self.prod_id = ''
        self.dir = dir
        self.data_dir = self.dir + "/dcm_data"
        self.database_dir = self.dir + "/dcm_database"
        self.train_dir = self.database_dir + "/train"
        self.temp_dir = self.database_dir + "/temp"
        self.model_dir = self.database_dir + "/model/single_class_classifier"
        self.checkpoint_dir = self.model_dir + "/checkpoint/single_class_classifier"
        self.tflite_dir = self.model_dir + "/tflite"

    def createTempDir(self):
        src_dir = os.path.join(self.train_dir, self.prod_id)
        dst_dir = os.path.join(self.temp_dir, self.prod_id)
        dst_ref_dir = os.path.join(self.temp_dir, "ref_dir")

        if not os.path.isdir(src_dir):
            print("{} doesn't exist".format(src_dir))
            exit()
        
        # Clean up temp dir if it exists
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except OSError as error:
            print(error)

        # copy /train/<product_dir> into /temp
        print('Copying', src_dir)
        shutil.copytree(src_dir, dst_dir)

        # Dynamic Negative Sampling
        # 1. Count positive samples
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        positive_files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_exts)]
        num_positives = len(positive_files)
        
        # 2. Collect all potential negative samples from other product folders
        negative_files = []
        all_products = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        
        print(f"Collecting negative samples from: {[p for p in all_products if p != self.prod_id]}")
        
        for prod in all_products:
            if prod == self.prod_id:
                continue
            
            prod_path = os.path.join(self.train_dir, prod)
            files = [os.path.join(prod_path, f) for f in os.listdir(prod_path) if f.lower().endswith(valid_exts)]
            negative_files.extend(files)
            
        # 3. Randomly sample to balance
        if len(negative_files) == 0:
            print("Error: No negative samples found in other product directories!")
            exit()
            
        if len(negative_files) < num_positives:
             print(f"Warning: Not enough negative samples ({len(negative_files)}) to fully balance positives ({num_positives}). Using all available.")
             selected_negatives = negative_files
        else:
             # Use numpy for random sampling
             selected_negatives = np.random.choice(negative_files, num_positives, replace=False)
             
        # 4. Copy negative samples to temp/ref_dir
        os.makedirs(dst_ref_dir, exist_ok=True)
        print(f'Creating balanced negative dataset: {len(selected_negatives)} images')
        
        for src in selected_negatives:
            # Create a unique name: sourcedirname_filename
            parent_folder = os.path.basename(os.path.dirname(src))
            filename = os.path.basename(src)
            new_name = f"{parent_folder}_{filename}"
            dst = os.path.join(dst_ref_dir, new_name)
            shutil.copy(src, dst)

    def removeTempDir(self):
        dst_dir = os.path.join(self.temp_dir, self.prod_id)
        dst_ref_dir = os.path.join(self.temp_dir, "ref_dir")
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        if os.path.exists(dst_ref_dir):
            shutil.rmtree(dst_ref_dir)

    def loadData(self):
        self.createTempDir()
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.temp_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE)
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.temp_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE)
        self.class_names = self.train_ds.class_names

    def loadModel(self, prod_id):
        self.prod_id = prod_id
        saved_dir = os.path.join(self.model_dir, self.prod_id)
        self.model = tf.keras.models.load_model(saved_dir)
        with open(os.path.join(saved_dir, 'class_names.names'), 'rb') as fp:
            self.class_names = pickle.load(fp)

    def loadModelQuantized(self, prod_id):
        self.prod_id = prod_id
        model_path = os.path.join(self.tflite_dir, self.prod_id+'_f16.tflite')
        with open(os.path.join(self.tflite_dir, self.prod_id+'.names'), 'rb') as fp:
            self.class_names = pickle.load(fp)

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def loadWeights(self):
        checkpoint = self.prod_id + ".cpkt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
        self.model.load_weights(checkpoint_path).expect_partial()

    def createModel(self, learning_rate=0.0005):
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)

        num_classes = len(self.class_names)
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
            tf.keras.layers.RandomTranslation(
                (-0.1, 0.1), (-0.1, 0.1)),
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        self.base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL),
                                                            include_top=False,
                                                            weights='imagenet')
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
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
            metrics=["accuracy"])
        self.model.summary()

    def finetuneModel(self, learning_rate=0.00005):
        self.base_model.trainable = True
        fine_tune_at = 100
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
            metrics=["accuracy"])
        self.model.summary()

    def trainModel(self, epochs, prev_history=None):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            initial_epoch=0 if prev_history == None else prev_history.epoch[-1]
        )
        return history

    def evaluateModel(self):
        loss, acc = self.model.evaluate(self.val_ds)
        print("Model accuracy: {:5.2f}%".format(100 * acc))
        return loss, acc

    def evaluateMetrics(self):
        print("\n--- Detailed Evaluation ---")
        y_true = []
        y_pred = []
        
        # Iterate over the validation dataset
        for images, labels in self.val_ds:
            predictions = self.model.predict(images, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predicted_labels)
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate Metrics
        accuracy = np.mean(y_true == y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"F1 Score (Weighted): {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
    def benchmarkInference(self, num_images=100):
        print("\n--- Inference Speed Benchmark ---")
        # Get a batch of images
        for images, _ in self.val_ds.take(1):
             sample_images = images
             break
        
        # Warmup
        print("Warming up...")
        _ = self.model.predict(sample_images, verbose=0)
        
        start_time = time.time()
        count = 0
        # Predict on random single images to simulate real-world inference
        # We use a loop to measure per-call overhead appropriately
        # Using the batch we fetched suitable for single prediction simulation
        
        # Flatten batch to list of single images
        single_images = [np.expand_dims(img, 0) for img in sample_images]
        
        # Run until we hit num_images or run out of batch
        limit = min(num_images, len(single_images))
        
        for i in range(limit):
            _ = self.model.predict(single_images[i], verbose=0)
            count += 1
            
        end_time = time.time()
        total_time = end_time - start_time
        
        avg_time_per_img = (total_time / count) * 1000 # in ms
        throughput = count / total_time # images/sec
        
        print(f"Inference Latency: {avg_time_per_img:.2f} ms/image")
        print(f"Throughput: {throughput:.2f} images/sec")
        
    def getModelSize(self, title):
        print("\n--- Model Size ---")
        # Check SavedModel directory size
        model_path = os.path.join(self.model_dir, title)
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        print(f"SavedModel Size: {total_size / (1024*1024):.2f} MB")
        
        # Check TFLite size
        tflite_path = os.path.join(self.tflite_dir, title + '_f16.tflite')
        if os.path.exists(tflite_path):
            tflite_size = os.path.getsize(tflite_path)
            print(f"TFloat16 TFLite Size: {tflite_size / (1024*1024):.2f} MB")
        else:
            print("TFLite model not found.")

    def saveModel(self, title):
        save_path = os.path.join(self.model_dir, title)
        self.model.save(save_path)
        print(f"Model saved to: {save_path}")

    def quantizeModel(self):
        os.makedirs(self.tflite_dir, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        
        with open(os.path.join(self.tflite_dir, self.prod_id+'_f16.tflite'), 'wb') as f:
            f.write(tflite_model)
        with open(os.path.join(self.tflite_dir, self.prod_id+'.names'), 'wb') as fp:
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

    def runTraining(self, epochs=100):
        initial_epochs = int(epochs/2)
        learning_rate = 0.0005

        self.loadData()
        self.createModel(learning_rate=learning_rate)
        history = self.trainModel(
            initial_epochs)
        self.evaluateModel()
        self.evaluateMetrics()
        self.finetuneModel(learning_rate=learning_rate/10)
        history_fine = self.trainModel(
            epochs, prev_history=history)
        self.evaluateModel()
        self.evaluateMetrics()
        self.saveModel(self.prod_id)
        self.quantizeModel()
        self.benchmarkInference()
        self.getModelSize(self.prod_id)
        self.removeTempDir()

    def runTrainingAll(self, epochs=100):
        tflite_files = [x for x in os.listdir(
            self.tflite_dir) if x.endswith('.tflite')]
        tflite_files = [x.partition('_')[0] for x in tflite_files]
        if not os.path.exists(self.train_dir):
            print("Train directory does not exist")
            return
            
        files = [x for x in os.listdir(self.train_dir) if x not in tflite_files]
        for f in files:
            if f[0] != ".":
                self.prod_id = f
                self.runTraining(epochs=epochs)

    def runFineTuning(self, epochs=50):
        learning_rate = 0.00005
        self.loadData()
        self.createModel()
        self.loadWeights()
        self.finetuneModel(learning_rate=learning_rate)
        history = self.trainModel(
            epochs)
        self.evaluateModel()
        self.evaluateMetrics()
        self.saveModel(self.prod_id)
        self.quantizeModel()
        self.benchmarkInference()
        self.getModelSize(self.prod_id)
        self.removeTempDir()

    def runHpTuning(self, epochs=10, finetune=False):
        self.loadData()
        learning_rates = [0.001, 0.0005, 0.0001, 0.00005]
        losses = []
        accuracies = []
        histories = []
        for lr in learning_rates:
            self.createModel(learning_rate=lr)
            if finetune:
                self.loadWeights()
                self.finetuneModel(learning_rate=lr)
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
        self.removeTempDir()
