#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
import tensorflow.keras as kr
import numpy as np

class TRNNConfig(object):

    embedding_dim = 64
    seq_length = 12
    num_classes = 14
    vocab_size = 1000

    num_layers= 1
    hidden_dim =64
    rnn = 'gru'

    dropout_keep_prob = 0.8
    learning_rate = 1e-3

    batch_size = 128
    num_epochs = 1

    print_per_batch = 100
    save_per_batch = 10


class TextRNN(object):
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.rnn()

    def rnn(self):
        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout():
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("rnn"):
            cells = [dropout() for _ in range(self.config.num_layers)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

            _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_inputs, dtype=tf.float32)
            last = _outputs[:, -1, :]

        with tf.name_scope("score"):
            fc = tf.layers.dense(last, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimize"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]
def train():

    f=open("train.txt")
    data=[]
    label=[]
    for line in f.readlines():
        arr=line.strip().split("\t")
        data.append(arr[1])
        label.append(int(arr[0]))
    f.close()
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = list(vocab_processor.fit_transform(data))
    y = kr.utils.to_categorical(label)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x, y, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                print "loss_train:" loss_train "\t" "acc_train:" acc_train

if __name__ == '__main__':
    config = TRNNConfig()
    model = TextRNN(config)
    train()
