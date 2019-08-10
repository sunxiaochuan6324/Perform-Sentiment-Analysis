from random import randint
import datetime
import numpy as np
import re
import tensorflow as tf
 
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
max_sequence_length = 30
numDimensions = 300
UNKNOWM = 399999
batch_size = 24
lstmUnits = 64
numClasses = 3
iterations = 100000
 
wordsList = np.load('wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')
 
 
def read_file(input_file):
    positive = []
    negative = []
    neutral = []
    with open(input_file, "r", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            label = line.split("\t")[1]
            text = line.split("\t")[2]
            if label == "positive":
                positive.append(text)
            elif label == "negative":
                negative.append(text)
            else:
                neutral.append(text)
        return positive, negative, neutral
 
 
def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())
 
 
def get_sentence_ids(sentence, ids, sentence_index):
    index_counter = 0
    cleaned_sentence = clean_sentences(sentence)
    split_sentence = cleaned_sentence.split()
    for word in split_sentence:
        try:
            ids[sentence_index][index_counter] = wordsList.index(word)
        except ValueError:
            ids[sentence_index][index_counter] = UNKNOWM
        index_counter = index_counter + 1
        if index_counter >= max_sequence_length:
            break
 
 
def get_ids(input_file, output_file):
    positive, negative, neutral = read_file(input_file)
    num_files = len(positive) + len(negative) + len(neutral)
    ids = np.zeros((num_files, max_sequence_length), dtype='int32')
    sentence_index = 0
    for sentence in positive:
        print("Processing new positive record", sentence_index)
        get_sentence_ids(sentence, ids, sentence_index)
        sentence_index = sentence_index + 1
 
    for sentence in negative:
        print("Processing new negative record", sentence_index)
        get_sentence_ids(sentence, ids, sentence_index)
        sentence_index = sentence_index + 1
 
    for sentence in neutral:
        print("Processing new neutral record", sentence_index)
        get_sentence_ids(sentence, ids, sentence_index)
        sentence_index = sentence_index + 1
 
    np.save(output_file, ids)
    print("Save successfully.")
 
 
def get_train_batch(ids):
    accurate_label = []
    array = np.zeros([batch_size, max_sequence_length])
    for index in range(batch_size):
        if index % 3 == 0:
            num = randint(1, 3646)
            accurate_label.append([1, 0, 0])
        elif index % 3 == 1:
            num = randint(3647, 5107)
            accurate_label.append([0, 1, 0])
        else:
            num = randint(5108, 9683)
            accurate_label.append([0, 0, 1])
        array[index] = ids[num - 1:num]
    return array, accurate_label
 
 
def get_validate_batch(ids):
    accurate_label = []
    array = np.zeros([batch_size, max_sequence_length])
    for index in range(batch_size):
        num = randint(1, 1654)
        if num <= 579:
            accurate_label.append([1, 0, 0])
        elif num <= 921:
            accurate_label.append([0, 1, 0])
        else:
            accurate_label.append([0, 0, 1])
        array[index] = ids[num - 1:num]
    return array, accurate_label
 
 
print("Converting raw data to idsMatrix...")
get_ids("train.txt", "train_idsMatrix.npy")
get_ids("dev.txt", "dev_idsMatrix.npy")
 
train_ids = np.load('train_idsMatrix.npy')
print("Load train_idsMatrix successfully. Start training...")
 
tf.reset_default_graph()
 
labels = tf.placeholder(tf.float32, [batch_size, numClasses])
input_data = tf.placeholder(tf.int32, [batch_size, max_sequence_length])
 
data = tf.Variable(tf.zeros([batch_size, max_sequence_length, numDimensions]), dtype=tf.float32)
 
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstm_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstm_cell, data, dtype=tf.float32)
 
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
 
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
 
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)
 
sess = tf.InteractiveSession()
saver = tf.train.Saver()
 
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)
 
sess.run(tf.global_variables_initializer())
 
for i in range(iterations):
    # Next Batch of reviews
    nextBatch, nextBatchLabels = get_train_batch(train_ids)
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
 
    # Write summary to Tensorboard
    if i % 50 == 0:
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)
 
    # Save the network every 10,000 training iterations
    if i % 1000 == 0 and i != 0:
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()
 
dev_ids = np.load('dev_idsMatrix.npy')
print("Load dev_idsMatrix successfully. Start validating...")
 
sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))
print("Load check point successfully.")
 
iterations = 100
total = 0
for i in range(iterations):
    print("i=", i, " ", end='')
    nextBatch, nextBatchLabels = get_validate_batch(dev_ids)
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
    total += (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
print("Average Accuracy:", total / iterations)