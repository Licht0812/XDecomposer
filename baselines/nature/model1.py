import random, shutil, glob

num_validation_file = 1000
test_validation_file = 1000
File_List = glob.glob('file path_total')
list_index = list(np.arange(0, len(File_List), 1))
random_index = random.sample(list_index,len(File_List))
for i in range(len(random_index)):
    random_index_ = random_index[i]
    print(random_index_)
#==========================================================================VALIDATION SET==========================================
    if i < num_validation_file:
        shutil.copy('file path_total' %random_index_, 'file path_Validation' %random_index_)
#==========================================================================Test SET==========================================
    if num_validation_file <= i and num_validation_file + test_validation_file > i:
        shutil.copy('file path_total' %random_index_, 'file path_Test' %random_index_)
#==========================================================================Training SET==========================================    
    if num_validation_file + test_validation_file <= i:
        shutil.copy('file path_total' %random_index_, 'file path_Training' %random_index_)       

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  

learning_rate = 0.001
num_files = 6010
vld_num_files = 1000
X_length = 4511
batch_size_ = 1000
n_classes = 38
epochs = 1

graph = tf.Graph()
with graph.as_default():
    
    filename_queue = tf.train.string_input_producer([('file path_Training' % i) for i in range(num_files)],  
                                                    shuffle=True, name='filename_queue')
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0] for _ in range(X_length)]
    record_defaults = [tf.constant([0], dtype=tf.float32) for _ in range(X_length)]
    xy_data = tf.decode_csv(value, record_defaults=record_defaults)
    xy_data = tf.stack(xy_data)
    y_1=tf.cast(xy_data[-7], tf.int32)
    y_2=tf.cast(xy_data[-6], tf.int32)
    y_3=tf.cast(xy_data[-5], tf.int32)
    y_1=tf.one_hot(y_1, n_classes)
    y_2=tf.one_hot(y_2, n_classes)
    y_3=tf.one_hot(y_3, n_classes)
    y_data = y_1 + y_2 + y_3
    y_data_ = y_1*xy_data[-4] + y_2*xy_data[-3] + y_3*xy_data[-2]
    y_data = tf.to_float(y_data)
    y_data_ = tf.to_float(y_data_)
    X_train, y_train,  y_train_, y_train_p, y_train_ind = tf.train.batch([xy_data[:-10], y_data, y_data_, xy_data[-4:-1],
                                                                xy_data[-1:]], batch_size = batch_size_)
    
#==========================================================================VALIDATION SET==========================================

    filename_queue_vld = tf.train.string_input_producer([('file path_Validation' % i) for i in range(vld_num_files)], 
                                                        shuffle=True, name='filename_queue')
    reader_vld = tf.TextLineReader()
    key, value_vld = reader.read(filename_queue)
    record_defaults = [[0] for _ in range(X_length)]
    record_defaults = [tf.constant([0], dtype=tf.float32) for _ in range(X_length)]
    xy_data_vld = tf.decode_csv(value_vld, record_defaults=record_defaults)
    xy_data_vld = tf.stack(xy_data_vld)
    y_1_vld=tf.cast(xy_data_vld[-7], tf.int32)
    y_2_vld=tf.cast(xy_data_vld[-6], tf.int32)
    y_3_vld=tf.cast(xy_data_vld[-5], tf.int32)
    y_1_vld=tf.one_hot(y_1_vld, n_classes)
    y_2_vld=tf.one_hot(y_2_vld, n_classes)
    y_3_vld=tf.one_hot(y_3_vld, n_classes)
    y_data_vld = y_1_vld + y_2_vld + y_3_vld
    y_data_vld_ = y_1_vld*xy_data_vld[-4] + y_2_vld*xy_data_vld[-3] + y_3_vld*xy_data_vld[-2]
    y_data_vld = tf.to_float(y_data_vld)
    y_data_vld_ = tf.to_float(y_data_vld_)
    X_vld, y_vld, y_vld_, y_vld_p, y_vld_ind = tf.train.batch([xy_data_vld[:-10], y_data_vld, y_data_vld_, xy_data_vld[-4:-1], 
                                                       xy_data_vld[-1:]], batch_size = batch_size_)   


with graph.as_default():
    
    inputs_ = tf.placeholder(tf.float32, [None, 4501, 1], name = 'inputs')
    labels_1 = tf.placeholder(tf.float32, [None, n_classes], name = 'labels_1')
    logit_num = tf.placeholder(tf.int32, [None, 3], name = 'logits_Top_3')
    label_num = tf.placeholder(tf.int32, [None, 3], name = 'labels_Top_3')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
    
    #   model architecture(CNN_2)    
"""
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=50, strides=2,padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=25, strides=3, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')
    flat = tf.reshape(max_pool_2, (-1, 126*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 500,  kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_2 = tf.layers.dense(logits_, n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
"""     

    #   model architecture(CNN_3)    
    
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=20, strides=1,padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=3, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=15, strides=1, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=2, padding='same',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    flat = tf.reshape(max_pool_3, (-1, 126*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2500, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 1000,  kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_2 = tf.layers.dense(logits_, n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
  
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_2, labels=labels_1))
    optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
    correct_pred = tf.equal(logit_num, label_num)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

validation_acc = []
validation_loss = []
train_acc = []
train_loss = []

with graph.as_default():
    saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('file path'))
    iteration = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for e in range(epochs):
        for i in  range(50):    
            X_tr, y_tr, y_tr_, y_tr_p, y_ind = sess.run([X_train, y_train, y_train_, y_train_p, y_train_ind])
            X_tr= np.reshape(X_tr, (-1, 4501, 1))
            feed = {inputs_ : X_tr, labels_1 : y_tr, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            loss, _ , logit = sess.run([cost, optimizer, logits_2], feed_dict = feed)            
            y_lab = np.empty([batch_size_, 3])
            y_logit = np.empty([batch_size_, 3])
            for i in range(batch_size_):
                if y_ind[i,0] == 2:
                    y_lab[i]=np.argsort(y_tr_[i])[-3:]
                    y_lab[i]=np.sort(y_lab[i])
                    y_logit[i]=np.argsort(logit[i])[-3:]
                    y_logit[i]=np.sort(y_logit[i])
                elif y_ind[i,0] == 1:
                    z=np.argsort(y_tr_[i])[-2:]
                    y_lab[i]=np.append(z, [0])
                    y_lab[i]=np.sort(y_lab[i])
                    z_=np.argsort(logit[i])[-2:]
                    y_logit[i]=np.append(z_, [0])
                    y_logit[i]=np.sort(y_logit[i])
                elif y_ind[i,0] == 0:
                    z=np.argsort(y_tr_[i])[-1:]
                    y_lab[i]=np.append(z, [0,0])
                    y_lab[i]=np.sort(y_lab[i])
                    z_=np.argsort(logit[i])[-1:]
                    y_logit[i]=np.append(z_, [0,0])
                    y_logit[i]=np.sort(y_logit[i])
            feed = {logit_num : y_logit, label_num: y_lab}
            acc = sess.run(accuracy, feed_dict = feed)
            train_acc.append(acc)
            train_loss.append(loss)                   
            print("Epoch: {}/{}".format(e, epochs),
                    "Iteration: {:d}".format(iteration),
                    "Train loss: {:6f}".format(loss),
                    "Train acc: {:.6f}".format(acc))

###================================================================ VALIDATION =====================================
            if (iteration %5 == 0):
                X_vd, y_vd, y_vd_, y_vd_p, y_ind_vd = sess.run([X_vld, y_vld, y_vld_, y_vld_p, y_vld_ind])
                X_vd= np.reshape(X_vd, (-1, 4501, 1))
                feed = {inputs_ : X_vd, labels_1 : y_vd, keep_prob_ : 1.0, learning_rate_ : learning_rate}
                loss_vd,  logit_vd = sess.run([cost, logits_2], feed_dict = feed)            
                y_lab_vd = np.empty([batch_size_, 3])
                y_logit_vd = np.empty([batch_size_, 3])
                for i in range(batch_size_):
                    if y_ind_vd[i,0] == 2:
                        y_lab_vd[i]=np.argsort(y_vd_[i])[-3:]
                        y_lab_vd[i]=np.sort(y_lab_vd[i])
                        y_logit_vd[i]=np.argsort(logit_vd[i])[-3:]
                        y_logit_vd[i]=np.sort(y_logit_vd[i])
                    elif y_ind_vd[i,0] == 1:
                        z=np.argsort(y_vd_[i])[-2:]
                        y_lab_vd[i]=np.append(z, [0])
                        y_lab_vd[i]=np.sort(y_lab_vd[i])
                        z_=np.argsort(logit_vd[i])[-2:]
                        y_logit_vd[i]=np.append(z_, [0])
                        y_logit_vd[i]=np.sort(y_logit_vd[i])
                    elif y_ind_vd[i,0] == 0:
                        z=np.argsort(y_vd_[i])[-1:]
                        y_lab_vd[i]=np.append(z, [0,0])
                        y_lab_vd[i]=np.sort(y_lab_vd[i])
                        z_=np.argsort(logit_vd[i])[-1:]
                        y_logit_vd[i]=np.append(z_, [0,0])
                        y_logit_vd[i]=np.sort(y_logit_vd[i])
                feed = {logit_num : y_logit_vd, label_num: y_lab_vd}
                acc_vd = sess.run(accuracy, feed_dict = feed)
                validation_acc.append(acc_vd)
                validation_loss.append(loss_vd)           
                print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {:d}".format(iteration),
                        "Validation loss: {:6f}".format(loss_vd),
                        "Validation acc: {:.6f}".format(acc_vd))                
            iteration += 1 
        saver.save(sess,"file path")
coord.request_stop()
coord.join(threads) 

