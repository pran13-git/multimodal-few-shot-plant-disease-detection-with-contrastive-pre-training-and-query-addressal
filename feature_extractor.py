import numpy as np
import tensorflow as tf
import cv2
from keras.models import Sequential, Model
from keras import layers
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, GlobalMaxPooling1D, Reshape, Dropout, Dense, Input, Concatenate, Lambda, MaxPooling1D, Flatten, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from transformers import BertTokenizer
import tensorflow.keras.backend as K
from transformers import TFBertModel

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def euclidean_distance(vectors):
  x, y = vectors
  sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))


def contrastive_loss(y_true, y_pred):
      margin=1
      square_pred = K.square(y_pred)
      margin_square = K.square(K.maximum(margin - y_pred, 0))
      y_true = K.cast(y_true, y_pred.dtype)
      return (0.5*y_true * square_pred + 0.5*(1 - y_true) * margin_square)

#@title Siamese Model

resnet_model = ResNet50(weights="imagenet", include_top=False)

def image_feat_network(input_shape):
    # Load ResNet50 model with pre-trained weights, excluding the top layer
    resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze all layers of the ResNet50 model
    for layer in resnet_model.layers:
        layer.trainable = False

    dense_layer = Dense(units=2048, activation='relu')(resnet_model.output)
    pooled_output = GlobalMaxPooling2D()(dense_layer)
    model = Model(inputs=resnet_model.input, outputs=pooled_output)

    return model

#@title Text Network

class ExtendedBert(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.bert = TFBertModel.from_pretrained("bert-base-cased",trainable=False)
        self.dense_layer = tf.keras.layers.Dense(units=2048)

    def call(self, inputs):
        input_ids = tf.cast(inputs['input_ids'], tf.int32)  # Cast input_ids to int32
        attention_mask = tf.cast(inputs['attention_mask'], tf.int32)
        token_type_ids = tf.cast(inputs['token_type_ids'], tf.int32)

        # get the hidden state of the last layer
        last_hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[0]
        first_token_hidden_state = last_hidden[:, 0, :]
        logits = self.dense_layer(first_token_hidden_state)
        return logits

def multi_modal_network(input_shape=(None, 4096)):
    seq = Sequential()
    seq.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    seq.add(MaxPooling1D(pool_size=2))
    seq.add(Conv1D(128, kernel_size=3, activation='relu'))
    seq.add(MaxPooling1D(pool_size=2))
    seq.add(Flatten())
    seq.add(Dense(256, activation='relu'))
    seq.add(Dropout(0.3))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.2))
    seq.add(Dense(50, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(2000, activation='relu'))

    return seq

def siamese_model(input_dim_img, input_dim_text, input_concat_dim):
    img_a = Input(shape=input_dim_img)
    img_b = Input(shape=input_dim_img)
    max_seq_length = 67

    text_a_input_ids = Input(shape=(max_seq_length,), name='input_ids')
    text_a_attention_mask = Input(shape=(max_seq_length,), name='attention_mask')
    text_a_token_type_ids = Input(shape=(max_seq_length,), name='token_type_ids')

    text_a = {'input_ids': text_a_input_ids,
              'attention_mask': text_a_attention_mask,
              'token_type_ids': text_a_token_type_ids}


    text_b_input_ids = Input(shape=(max_seq_length,), name='input_ids_b')
    text_b_attention_mask = Input(shape=(max_seq_length,), name='attention_mask_b')
    text_b_token_type_ids = Input(shape=(max_seq_length,), name='token_type_ids_b')


    text_b = {'input_ids': text_b_input_ids,
              'attention_mask': text_b_attention_mask,
              'token_type_ids': text_b_token_type_ids}

    img_network = image_feat_network(input_dim_img)
    text_network = ExtendedBert()

    feat_img_a = img_network(img_a)
    feat_img_b = img_network(img_b)

    text_a_tf = {key: tf.convert_to_tensor(value) for key, value in text_a.items()}
    text_b_tf = {key: tf.convert_to_tensor(value) for key, value in text_b.items()}

    feat_text_a = text_network(text_a_tf)
    feat_text_b = text_network(text_b_tf)

    concat_a = Concatenate(axis=-1)([feat_img_a, feat_text_a])
    concat_b = Concatenate(axis=-1)([feat_img_b, feat_text_b])


    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([concat_a, concat_b])
    prediction = distance

    model = Model(inputs=[img_a, text_a_input_ids, text_a_attention_mask, text_a_token_type_ids,
                          img_b, text_b_input_ids, text_b_attention_mask, text_b_token_type_ids],
                  outputs=prediction)
    rms = tf.keras.optimizers.Adam(learning_rate=0.0001)

    return rms, model

def process_text_input(text_inputs):

  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

  # Tokenize the texts
  tokenized_inputs = tokenizer(text_inputs, padding='max_length', truncation=True, max_length=67, return_tensors='tf')

  input_ids = tokenized_inputs['input_ids']
  attention_mask = tokenized_inputs['attention_mask']
  token_type_ids = tokenized_inputs.get('token_type_ids')


  max_seq_length = 67
  batch_size = 1

  input_ids = tf.reshape(input_ids, (batch_size, max_seq_length))
  attention_mask = tf.reshape(attention_mask, (batch_size, max_seq_length))
  token_type_ids = tf.reshape(token_type_ids, (batch_size, max_seq_length))


  input_ids = tf.cast(input_ids, tf.int32)
  attention_mask = tf.cast(attention_mask, tf.float32)
  token_type_ids = tf.cast(token_type_ids, tf.int32)


  text = {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'token_type_ids': token_type_ids
  }

  return text

def process_input(image_path, description):
  image = cv2.imread(image_path)
  image = cv2.resize(image, (64, 64))
  image = image / 255.0
  img_data = np.asarray(image, dtype=np.float32)
  img_data = np.expand_dims(img_data, axis=0)
  img_tensor = tf.convert_to_tensor(img_data)
  image_lst = tf.convert_to_tensor(list(img_data))

  text = process_text_input(description)


  return image_lst, text

def init_siamese(ckpt_path):
  MAX_LEN = 67
  INPUT_DIM_IMG = (64, 64, 3)
  INPUT_DIM_TEXT = (67,)
  INPUT_CONCAT_DIM = 33

  opt,loaded_model = siamese_model (INPUT_DIM_IMG ,INPUT_DIM_TEXT , INPUT_CONCAT_DIM)
  loaded_model.load_weights(ckpt_path)

  model_siamese = Model(inputs=loaded_model.input, outputs=loaded_model.layers[-1].input)

  return model_siamese

def predict_siamese(url, desc, model):
  image, text = process_input(url, desc)
  op = model.predict([image,
                    text['input_ids'],text['attention_mask'],text['token_type_ids'],
                    image,
                    text['input_ids'],text['attention_mask'],text['token_type_ids']])

  return op[0]