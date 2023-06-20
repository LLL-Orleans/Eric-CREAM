import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  LSTM, Dropout, Dense
from data_tf import Dataset

def cos_sim(x, y):
    x_dot_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
    x_dot_x = tf.reduce_sum(tf.square(x), axis=1)
    y_dot_y = tf.reduce_sum(tf.square(y), axis=1)
    return tf.divide(x_dot_y, tf.multiply(tf.sqrt(x_dot_x), tf.sqrt(y_dot_y)))


def triplet_hinge(anchor, same, diff, margin):
    return tf.maximum(0., margin + cos_sim(anchor, diff) - cos_sim(anchor, same))


def triplet_loss(logits, same_partition, diff_partition, margin=0.3):
    logits = tf.math.segment_mean(logits, same_partition)
    batch_size = tf.math.reduce_max(diff_partition) + 1
    anchor, same, diff = tf.split(logits, [batch_size, batch_size, -1])

    
    anchor = tf.gather(anchor, diff_partition)
    same = tf.gather(same, diff_partition)
    losses = triplet_hinge(anchor, same, diff, margin)
    losses = tf.math.segment_max(losses, diff_partition)

    return tf.math.reduce_mean(losses)

loss = triplet_loss

class Classifier(tf.keras.Model):
    def __init__(self, input_dim, hidden_units=512, dropout=0.2, recurrent_dropout=0.2):
        super(Classifier, self).__init__()

        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.lstm1 = LSTM(self.hidden_units, dropout=0.25,return_sequences=True)
        self.lstm2 = LSTM(self.hidden_units, dropout=0.25,return_sequences=True)
        self.lstm3 = LSTM(self.hidden_units, dropout=0.25,return_sequences=True)
        self.lstm4 = LSTM(self.hidden_units, dropout=0.25,return_sequences=True)
        self.lstm5 = LSTM(self.hidden_units, dropout=0.25)


    def call(self, inp):
        #change to x1 x2 if bug

        x = self.lstm1(inp)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.lstm5(x)
     
        return x

def grad(model, x, x_len, same_partition, diff_partition):
    with tf.GradientTape() as tape:
        output = model(x)
        loss_value = loss(output, same_partition, diff_partition)
        gradient = tape.gradient(loss_value, model.trainable_variables)

        grads = [tf.clip_by_norm(g, 2.0) for g in gradient]
    return loss_value, grads, x, output


def main():

    train_path = "/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/trainWords/"
    dev_path = "/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/devWords/"

    train_data = Dataset(wav_path=train_path, partition="train")
    dev_data = Dataset(wav_path=dev_path, partition="dev")
    classifier = Classifier(39)
    
    optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    global_step = tf.Variable(0)

    losses = []
    for epoch in range(25):
        print("Epoch: ", epoch)

        batch_id = 0
        for x, x_len, same, diff in train_data.batch(batch_size=16, max_same=1, max_diff=1):
            print(x.shape)
            loss_value, grads, inputs, reconstruction = grad(classifier, x, x_len, same, diff)
            optimizer.apply_gradients(zip(grads, classifier.trainable_variables),
                                global_step)
            print("batch nb. {}      loss: {}".format(batch_id, loss_value))
            batch_id +=1

main()