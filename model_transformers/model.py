# Script to test the model in AWS


from tensorflow.keras import layers
from tensorflow import keras

from tqdm import tqdm
from time import sleep
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

# ---------------------- Configs -------------------------

MAX_SEQ_LENGTH = 8
NUM_FEATURES = 2048
IMG_SIZE = 224
EPOCHS = 3

train_df = pd.read_csv("finish_data_train.csv")
test_df = pd.read_csv("finish_data_test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

center_crop_layer = layers.CenterCrop(IMG_SIZE, IMG_SIZE)

# ----------------------------------------------------------

def crop_center(frame):
    cropped = center_crop_layer(frame[None, ...])
    cropped = cropped.numpy().squeeze()
    return cropped

def load_video(path, max_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(path)
    
    # Total frames
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = round(length/max_frames)
    
    total = 0
    frames = []
    i=0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or total == max_frames:
                break
            else:
                pass
            
            if i%n_frames == 0:
                frame = crop_center(frame)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)
                total += 1
            
            i+=1
            
        if total < max_frames:
            cap = cv2.VideoCapture(path)
            for j in range(int(length)):
                ret, frame = cap.read()
                if j == (int(length)-1):
                    frame = crop_center(frame)
                    frame = frame[:, :, [2, 1, 0]]
                    frames.append(frame)
            
    finally:
        cap.release()
    return np.array(frames)

# ----------------------------------------------------------

def build_feature_extractor():
    feature_extractor = keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.xception.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


# ----------------------------------------------------------

# Xception model
feature_extractor = build_feature_extractor()

feature_extractor.summary()


# Label preprocessing with StringLookup.
label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df["movement_label"]), mask_token=None
)
print("Predicción de clases:")
print(label_processor.get_vocabulary())


# ----------------------------------------------------------

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["dir"].values.tolist()
    labels = df["movement_label"].values
    labels = label_processor(labels[..., None]).numpy()

    # `frame_features` are what we will feed to our sequence model.
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in tqdm(enumerate(video_paths), desc='Processing T:15844/196'):
        # Gather all its frames and add a batch dimension.
        frames = load_video(path)

        frames = frames[None, ...]

        # Initialize placeholder to store the features of the current video.
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                if np.mean(batch[j, :]) > 0.0:
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )

                else:
                    temp_frame_features[i, j, :] = 0.0

        frame_features[idx,] = temp_frame_features.squeeze()

    return frame_features, labels


# ----------------------------------------------------------

print("--------- Prepare all video -----------")
train_data, train_labels = prepare_all_videos(train_df, "train")
test_data, test_labels = prepare_all_videos(test_df, "test")

try:
    
    np.save("train_data.npy", train_data)
    print("Save OK 1/4")
    np.save("train_labels.npy", train_labels)
    print("Save OK 2/4")
    np.save("test_data.npy", test_data)
    print("Save OK 3/4")
    np.save("test_labels.npy", test_labels)
    print("Save OK 4/4")
    
except:
    print("File error save .npy")
    pass

#train_data, train_labels = np.load("train_data.npy"), np.load("train_labels.npy")
#test_data, test_labels = np.load("test_data.npy"), np.load("test_labels.npy")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")

# ----------------------------------------------------------


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def call(self, inputs):
        # The inputs are of shape: `(batch_size, frames, num_features)`
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def compute_mask(self, inputs, mask=None):
        mask = tf.reduce_any(tf.cast(inputs, "bool"), axis=-1)
        return mask

# ----------------------------------------------------------

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

# ----------------------------------------------------------

def get_compiled_model():
    sequence_length = MAX_SEQ_LENGTH
    embed_dim = NUM_FEATURES
    dense_dim = 4
    num_heads = 1
    classes = len(label_processor.get_vocabulary())

    inputs = keras.Input(shape=(None, None))

    x = PositionalEmbedding(sequence_length, embed_dim, name="frame_position_embedding")(inputs)

    x = TransformerEncoder(embed_dim, dense_dim, num_heads, name="transformer_layer")(x)
    
    x = layers.GlobalMaxPooling1D(data_format="channels_last", keepdims=False,)(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, epsilon=0.1), 
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        metrics=["accuracy"]
    )
    return model

# ----------------------------------------------------------


def run_experiment():
    filepath = "/tmp/video_classifier"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    model = get_compiled_model()
    history = model.fit(
        train_data,
        train_labels,
        validation_split=0.15,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    model.load_weights(filepath)
    _, accuracy = model.evaluate(test_data, test_labels)
    t = open("accu_test.txt", "wt")
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    t.write(f"Test accuracy: {round(accuracy * 100, 2)}%")
    t.close()
    return model

# ----------------------------------------------------------

print("--- RUN EXPERIMENT ---")
trained_model = run_experiment()

try:
    trained_model.save('my_model_alex.h5')
    print("--- Model Save ---")

except:
    pass

# ----------------------------------------------------------

print("---- PREDICTIONS -----")
def prepare_single_video(frames):
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    # Pad shorter videos.
    if len(frames) < MAX_SEQ_LENGTH:
        diff = MAX_SEQ_LENGTH - len(frames)
        padding = np.zeros((diff, IMG_SIZE, IMG_SIZE, 3))
        frames = np.concatenate(frames, padding)

    frames = frames[None, ...]

    # Extract features from the frames of the current video.
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            if np.mean(batch[j, :]) > 0.0:
                frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
            else:
                frame_features[i, j, :] = 0.0

    return frame_features


# ----------------------------------------------------------

def predict_action(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(path)
    frame_features = prepare_single_video(frames)
    probabilities = trained_model.predict(frame_features)[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


print("Prediction:")
    
test_video = np.random.choice(test_df["dir"].values.tolist())
print(f"Test video path: {test_video}")    
test_frames = predict_action(test_video)


# ----------------------------------------------------------