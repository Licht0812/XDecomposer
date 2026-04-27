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
num_files_te = 1000
vld_num_files = 1000
X_length = 4511
batch_size_ = 1000
n_classes = 38
epochs = 1

graph = tf.Graph()
with graph.as_default():
    
    filename_queue = tf.train.string_input_producer([('file path' % i) for i in range(num_files)], 
                                                    shuffle=True, name='filename_queue')
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)
    record_defaults = [[0] for _ in range(X_length)]
    record_defaults = [tf.constant([0], dtype=tf.float32) for _ in range(X_length)]
    xy_data = tf.decode_csv(value, record_defaults=record_defaults)
    xy_data = tf.stack(xy_data)
   
    A_1=tf.cast(xy_data[-4], tf.float32)
    A_2=tf.cast(xy_data[-3], tf.float32)
    A_3=tf.cast(xy_data[-2], tf.float32)
    y_1=tf.cast(xy_data[-7], tf.int32)
    y_2=tf.cast(xy_data[-6], tf.int32)
    y_3=tf.cast(xy_data[-5], tf.int32)

    # 3_Level_Fraction_Prediction(Training)    
    y_1 = tf.cond(tf.greater_equal(A_1, 0.6667), lambda:y_1+76, lambda:y_1)
    y_1 = tf.cond(tf.greater_equal(A_1, 0.3333) & tf.less(A_1, 0.6667), lambda:y_1+38, lambda:y_1)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.6667), lambda:y_2+76, lambda:y_2)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.3333) & tf.less(A_2, 0.6667), lambda:y_2+38, lambda:y_2)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.6667), lambda:y_3+76, lambda:y_3)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.3333) & tf.less(A_3, 0.6667), lambda:y_3+38, lambda:y_3)
    y_1=tf.one_hot(y_1, n_classes*3)
    y_2=tf.one_hot(y_2, n_classes*3)
    y_3=tf.one_hot(y_3, n_classes*3)
    
    # 4_Level_Fraction_Prediction(Training)  
"""
    y_1 = tf.cond(tf.greater_equal(A_1, 0.75), lambda:y_1+114, lambda:y_1)
    y_1 = tf.cond(tf.greater_equal(A_1, 0.5) & tf.less(A_1, 0.75), lambda:y_1+76, lambda:y_1)
    y_1 = tf.cond(tf.greater_equal(A_1, 0.25) & tf.less(A_1, 0.5), lambda:y_1+38, lambda:y_1)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.75), lambda:y_2+114, lambda:y_2)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.5) & tf.less(A_2, 0.75), lambda:y_2+76, lambda:y_2)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.25) & tf.less(A_2, 0.5), lambda:y_2+38, lambda:y_2)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.75), lambda:y_3+114, lambda:y_3)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.5) & tf.less(A_3, 0.75), lambda:y_3+76, lambda:y_3)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.25) & tf.less(A_3, 0.5), lambda:y_3+38, lambda:y_3)
    y_1=tf.one_hot(y_1, n_classes*4)
    y_2=tf.one_hot(y_2, n_classes*4)
    y_3=tf.one_hot(y_3, n_classes*4)
"""    
    # 5_Level_Fraction_Prediction(Training)
"""    
    y_1 = tf.cond(tf.greater_equal(A_1, 0.8), lambda:y_1+152, lambda:y_1)
    y_1 = tf.cond(tf.greater_equal(A_1, 0.6) & tf.less(A_1, 0.8), lambda:y_1+114, lambda:y_1)
    y_1 = tf.cond(tf.greater_equal(A_1, 0.4) & tf.less(A_1, 0.6), lambda:y_1+76, lambda:y_1)
    y_1 = tf.cond(tf.greater_equal(A_1, 0.2) & tf.less(A_1, 0.4), lambda:y_1+38, lambda:y_1)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.8), lambda:y_2+152, lambda:y_2)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.6) & tf.less(A_2, 0.8), lambda:y_2+114, lambda:y_2)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.4) & tf.less(A_2, 0.6), lambda:y_2+76, lambda:y_2)
    y_2 = tf.cond(tf.greater_equal(A_2, 0.2) & tf.less(A_2, 0.4), lambda:y_2+38, lambda:y_2)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.8), lambda:y_3+152, lambda:y_3)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.6) & tf.less(A_3, 0.8), lambda:y_3+114, lambda:y_3)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.4) & tf.less(A_3, 0.6), lambda:y_3+76, lambda:y_3)
    y_3 = tf.cond(tf.greater_equal(A_3, 0.2) & tf.less(A_3, 0.4), lambda:y_3+38, lambda:y_3)
    y_1=tf.one_hot(y_1, n_classes*5)
    y_2=tf.one_hot(y_2, n_classes*5)
    y_3=tf.one_hot(y_3, n_classes*5)
"""    
    y_data = y_1 + y_2 + y_3  
    y_data = tf.to_float(y_data)
    X_train, y_train,  y_train_p, y_train_ind = tf.train.batch([xy_data[:-10], y_data,  xy_data[-4:-1],
                                                                xy_data[-1:]], batch_size = batch_size_)
    y_train_p, _ = tf.nn.top_k(y_train_p, k=3, sorted=True)
    y_train_p = tf.reshape(y_train_p, [batch_size_, 3])

#==========================================================================VALIDATION SET==========================================

    filename_queue_vld = tf.train.string_input_producer([('file path' % i) for i in range(vld_num_files)],
                                                        shuffle=True, name='filename_queue')
    reader_vld = tf.TextLineReader()
    key, value_vld = reader.read(filename_queue)
    record_defaults = [[0] for _ in range(X_length)]
    record_defaults = [tf.constant([0], dtype=tf.float32) for _ in range(X_length)]
    xy_data_vld = tf.decode_csv(value_vld, record_defaults=record_defaults)
    xy_data_vld = tf.stack(xy_data_vld)
      
    A_1_vld=tf.cast(xy_data_vld[-4], tf.float32)
    A_2_vld=tf.cast(xy_data_vld[-3], tf.float32)
    A_3_vld=tf.cast(xy_data_vld[-2], tf.float32)
    y_1_vld=tf.cast(xy_data_vld[-7], tf.int32)
    y_2_vld=tf.cast(xy_data_vld[-6], tf.int32)
    y_3_vld=tf.cast(xy_data_vld[-5], tf.int32)

    # 3_Level_Fraction_Prediction(Validation) 
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.6667), lambda:y_1_vld+76, lambda:y_1_vld)
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.3333) & tf.less(A_1_vld, 0.6667), lambda:y_1_vld+38, lambda:y_1_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.6667), lambda:y_2_vld+76, lambda:y_2_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.3333) & tf.less(A_2_vld, 0.6667), lambda:y_2_vld+38, lambda:y_2_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.6667), lambda:y_3_vld+76, lambda:y_3_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.3333) & tf.less(A_3_vld, 0.6667), lambda:y_3_vld+38, lambda:y_3_vld)
    y_1_vld=tf.one_hot(y_1_vld, n_classes*3)
    y_2_vld=tf.one_hot(y_2_vld, n_classes*3)
    y_3_vld=tf.one_hot(y_3_vld, n_classes*3)
   
    # 4_Level_Fraction_Prediction(Validation)  
"""
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.75), lambda:y_1_vld+114, lambda:y_1_vld)
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.5) & tf.less(A_1_vld, 0.75), lambda:y_1_vld+76, lambda:y_1_vld)
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.25) & tf.less(A_1_vld, 0.5), lambda:y_1_vld+38, lambda:y_1_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.75), lambda:y_2_vld+114, lambda:y_2_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.5) & tf.less(A_2_vld, 0.75), lambda:y_2_vld+76, lambda:y_2_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.25) & tf.less(A_2_vld, 0.5), lambda:y_2_vld+38, lambda:y_2_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.75), lambda:y_3_vld+114, lambda:y_3_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.5) & tf.less(A_3_vld, 0.75), lambda:y_3_vld+76, lambda:y_3_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.25) & tf.less(A_3_vld, 0.5), lambda:y_3_vld+38, lambda:y_3_vld)
    y_1_vld=tf.one_hot(y_1_vld, n_classes*4)
    y_2_vld=tf.one_hot(y_2_vld, n_classes*4)
    y_3_vld=tf.one_hot(y_3_vld, n_classes*4)
"""    
    # 5_Level_Fraction_Prediction(Validation)
"""    
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.8), lambda:y_1_vld+152, lambda:y_1_vld)
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.6) & tf.less(A_1_vld, 0.8), lambda:y_1_vld+114, lambda:y_1_vld)
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.4) & tf.less(A_1_vld, 0.6), lambda:y_1_vld+76, lambda:y_1_vld)
    y_1_vld = tf.cond(tf.greater_equal(A_1_vld, 0.2) & tf.less(A_1_vld, 0.4), lambda:y_1_vld+38, lambda:y_1_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.8), lambda:y_2_vld+152, lambda:y_2_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.6) & tf.less(A_2_vld, 0.8), lambda:y_2_vld+114, lambda:y_2_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.4) & tf.less(A_2_vld, 0.6), lambda:y_2_vld+76, lambda:y_2_vld)
    y_2_vld = tf.cond(tf.greater_equal(A_2_vld, 0.2) & tf.less(A_2_vld, 0.4), lambda:y_2_vld+38, lambda:y_2_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.8), lambda:y_3_vld+152, lambda:y_3_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.6) & tf.less(A_3_vld, 0.8), lambda:y_3_vld+114, lambda:y_3_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.4) & tf.less(A_3_vld, 0.6), lambda:y_3_vld+76, lambda:y_3_vld)
    y_3_vld = tf.cond(tf.greater_equal(A_3_vld, 0.2) & tf.less(A_3_vld, 0.4), lambda:y_3_vld+38, lambda:y_3_vld)
    y_1_vld=tf.one_hot(y_1_vld, n_classes*5)
    y_2_vld=tf.one_hot(y_2_vld, n_classes*5)
    y_3_vld=tf.one_hot(y_3_vld, n_classes*5)
"""    
    y_data_vld = y_1_vld + y_2_vld + y_3_vld
    y_data_vld = tf.to_float(y_data_vld)
    X_vld, y_vld, y_vld_p, y_vld_ind = tf.train.batch([xy_data_vld[:-10], y_data_vld, xy_data_vld[-4:-1], 
                                                       xy_data_vld[-1:]], batch_size = batch_size_)
    y_vld_p, _ = tf.nn.top_k(y_vld_p, k=3, sorted=True)
    y_vld_p = tf.reshape(y_vld_p, [batch_size_, 3])


with graph.as_default():
    
    inputs_ = tf.placeholder(tf.float32, [None, 4501, 1], name = 'inputs')
    labels_1 = tf.placeholder(tf.float32, [None, n_classes*3], name = 'labels_1')
    labels_2 = tf.placeholder(tf.float32, [None, 3], name = 'labels_2')
    logit_num = tf.placeholder(tf.int32, [None, 3], name = 'logits_Top_3')
    label_num = tf.placeholder(tf.int32, [None, 3], name = 'labels_Top_3')
    keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
    learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')

    #   model architecture(CNN_2F)        
""""" 
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=50, strides=2,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=25, strides=3, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')
    flat = tf.reshape(max_pool_2, (-1, 126*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 500, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*3, kernel_initializer=tf.contrib.layers.xavier_initializer())
"""
    #   model architecture(CNN_3F)
"""    
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=20, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=3, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=15, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    flat = tf.reshape(max_pool_3, (-1, 126*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2500, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 1000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*3, kernel_initializer=tf.contrib.layers.xavier_initializer())
"""
    #   model architecture(CNN_4F)    
"""
    conv = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=25, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=3, strides=2, padding='same')
    conv1 = tf.layers.conv1d(inputs=max_pool, filters=64, kernel_size=20, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=15, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    flat = tf.reshape(max_pool_3, (-1, 141*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2500, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 1000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*3, kernel_initializer=tf.contrib.layers.xavier_initializer())
""" 
    #   model architecture(CNN_5F)      

    conv = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=30, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=3, strides=2, padding='same')
    conv1 = tf.layers.conv1d(inputs=max_pool, filters=64, kernel_size=25, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=20, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=15, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=64, kernel_size=10, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=1, strides=2, padding='same')
    flat = tf.reshape(max_pool_4, (-1, 141*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2500, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 1000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*3, kernel_initializer=tf.contrib.layers.xavier_initializer())
 
    #   model architecture(CNN_6F) 
"""    
    conv = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=35, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool = tf.layers.max_pooling1d(inputs=conv, pool_size=3, strides=2, padding='same')
    conv1 = tf.layers.conv1d(inputs=max_pool, filters=64, kernel_size=30, strides=1,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=2, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=25, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=20, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=64, kernel_size=15, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=1, strides=2, padding='same')
    conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=64, kernel_size=10, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_5 = tf.layers.max_pooling1d(inputs=conv4, pool_size=1, strides=2, padding='same')
    flat = tf.reshape(max_pool_5, (-1, 71*64))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 2500, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 1000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*3, kernel_initializer=tf.contrib.layers.xavier_initializer())
"""
    #   model architecture(CNN_3I)
""" 
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=20, strides=2,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=3, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=15, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    conv1_11 = tf.layers.conv1d(inputs=max_pool_3, filters=22, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv1_21 = tf.layers.conv1d(inputs=max_pool_3, filters=32, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv1_31 = tf.layers.conv1d(inputs=max_pool_3, filters=6, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool1_41 = tf.layers.average_pooling1d(inputs=max_pool_3, pool_size=3, strides=1, padding='same')
    conv1_22 = tf.layers.conv1d(inputs=conv1_21, filters=42, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv1_32 = tf.layers.conv1d(inputs=conv1_31, filters=12, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv1_42 = tf.layers.conv1d(inputs=avg_pool1_41, filters=10, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_1 = tf.concat([conv1_11, conv1_22, conv1_32, conv1_42], axis=2) 
    conv2_11 = tf.layers.conv1d(inputs=inception_out_1, filters=28, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv2_21 = tf.layers.conv1d(inputs=inception_out_1, filters=43, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv2_31 = tf.layers.conv1d(inputs=inception_out_1, filters=7, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool2_41 = tf.layers.average_pooling1d(inputs=inception_out_1, pool_size=3, strides=1, padding='same')
    conv2_22 = tf.layers.conv1d(inputs=conv2_21, filters=56, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv2_32 = tf.layers.conv1d(inputs=conv2_31, filters=14, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv2_42 = tf.layers.conv1d(inputs=avg_pool2_41, filters=14, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_2 = tf.concat([conv2_11, conv2_22, conv2_32, conv2_42], axis=2) 
    conv3_11 = tf.layers.conv1d(inputs=inception_out_2, filters=37, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv3_21 = tf.layers.conv1d(inputs=inception_out_2, filters=56, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv3_31 = tf.layers.conv1d(inputs=inception_out_2, filters=9, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool3_41 = tf.layers.average_pooling1d(inputs=inception_out_2, pool_size=3, strides=1, padding='same')
    conv3_22 = tf.layers.conv1d(inputs=conv3_21, filters=73, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv3_32 = tf.layers.conv1d(inputs=conv3_31, filters=18, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv3_42 = tf.layers.conv1d(inputs=avg_pool3_41, filters=19, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_3 = tf.concat([conv3_11, conv3_22, conv3_32, conv3_42], axis=2)
    flat = tf.reshape(inception_out_3, (-1, 126*147))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 3700, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 740, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*5, kernel_initializer=tf.contrib.layers.xavier_initializer())
"""    
    #   model architecture(CNN_6I)
"""
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=64, kernel_size=20, strides=2,padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=3, strides=3, padding='same')
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=15, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=3, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=64, kernel_size=10, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), activation = tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=1, strides=2, padding='same')
    conv1_11 = tf.layers.conv1d(inputs=max_pool_3, filters=22, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv1_21 = tf.layers.conv1d(inputs=max_pool_3, filters=32, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv1_31 = tf.layers.conv1d(inputs=max_pool_3, filters=6, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool1_41 = tf.layers.average_pooling1d(inputs=max_pool_3, pool_size=3, strides=1, padding='same')
    conv1_22 = tf.layers.conv1d(inputs=conv1_21, filters=42, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv1_32 = tf.layers.conv1d(inputs=conv1_31, filters=12, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv1_42 = tf.layers.conv1d(inputs=avg_pool1_41, filters=10, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_1 = tf.concat([conv1_11, conv1_22, conv1_32, conv1_42], axis=2) 
    conv2_11 = tf.layers.conv1d(inputs=inception_out_1, filters=28, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv2_21 = tf.layers.conv1d(inputs=inception_out_1, filters=43, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv2_31 = tf.layers.conv1d(inputs=inception_out_1, filters=7, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool2_41 = tf.layers.average_pooling1d(inputs=inception_out_1, pool_size=3, strides=1, padding='same')
    conv2_22 = tf.layers.conv1d(inputs=conv2_21, filters=56, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv2_32 = tf.layers.conv1d(inputs=conv2_31, filters=14, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv2_42 = tf.layers.conv1d(inputs=avg_pool2_41, filters=14, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_2 = tf.concat([conv2_11, conv2_22, conv2_32, conv2_42], axis=2) 
    conv3_11 = tf.layers.conv1d(inputs=inception_out_2, filters=37, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv3_21 = tf.layers.conv1d(inputs=inception_out_2, filters=56, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv3_31 = tf.layers.conv1d(inputs=inception_out_2, filters=9, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool3_41 = tf.layers.average_pooling1d(inputs=inception_out_2, pool_size=3, strides=1, padding='same')
    conv3_22 = tf.layers.conv1d(inputs=conv3_21, filters=73, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv3_32 = tf.layers.conv1d(inputs=conv3_31, filters=18, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv3_42 = tf.layers.conv1d(inputs=avg_pool3_41, filters=19, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_3 = tf.concat([conv3_11, conv3_22, conv3_32, conv3_42], axis=2)
    conv4_11 = tf.layers.conv1d(inputs=inception_out_3, filters=49, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv4_21 = tf.layers.conv1d(inputs=inception_out_3, filters=74, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv4_31 = tf.layers.conv1d(inputs=inception_out_3, filters=12, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool4_41 = tf.layers.average_pooling1d(inputs=inception_out_3, pool_size=3, strides=1, padding='same')
    conv4_22 = tf.layers.conv1d(inputs=conv4_21, filters=96, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv4_32 = tf.layers.conv1d(inputs=conv4_31, filters=24, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv4_42 = tf.layers.conv1d(inputs=avg_pool4_41, filters=25, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_4 = tf.concat([conv4_11, conv4_22, conv4_32, conv4_42], axis=2)
    conv5_11 = tf.layers.conv1d(inputs=inception_out_4, filters=65, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv5_21 = tf.layers.conv1d(inputs=inception_out_4, filters=97, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv5_31 = tf.layers.conv1d(inputs=inception_out_4, filters=16, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool5_41 = tf.layers.average_pooling1d(inputs=inception_out_4, pool_size=3, strides=1, padding='same')
    conv5_22 = tf.layers.conv1d(inputs=conv4_21, filters=126, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv5_32 = tf.layers.conv1d(inputs=conv4_31, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv5_42 = tf.layers.conv1d(inputs=avg_pool4_41, filters=32, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_5 = tf.concat([conv5_11, conv5_22, conv5_32, conv5_42], axis=2)
    conv6_11 = tf.layers.conv1d(inputs=inception_out_5, filters=85, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv6_21 = tf.layers.conv1d(inputs=inception_out_5, filters=128, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    conv6_31 = tf.layers.conv1d(inputs=inception_out_5, filters=19, kernel_size=1, strides=1, padding='same', activation = tf.nn.relu)
    avg_pool6_41 = tf.layers.average_pooling1d(inputs=inception_out_5, pool_size=3, strides=1, padding='same')
    conv6_22 = tf.layers.conv1d(inputs=conv6_21, filters=166, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
    conv6_32 = tf.layers.conv1d(inputs=conv6_31, filters=38, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    conv6_42 = tf.layers.conv1d(inputs=avg_pool6_41, filters=43, kernel_size=1, strides=1, padding='same', activation=tf.nn.relu)
    inception_out_6 = tf.concat([conv6_11, conv6_22, conv6_32, conv6_42], axis=2)
    flat = tf.reshape(inception_out_6, (-1, 126*332))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, 4000, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits= tf.nn.dropout(logits, keep_prob=keep_prob_)
    logits_ = tf.layers.dense(logits, 400, kernel_initializer=tf.contrib.layers.xavier_initializer())
    logits_= tf.nn.dropout(logits_, keep_prob=keep_prob_)
    logits_1 = tf.layers.dense(logits_, n_classes*5, kernel_initializer=tf.contrib.layers.xavier_initializer())
""""    
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_1,labels=labels_1)) 
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
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    iteration = 1
    for e in range(epochs):
        for i in  range(600):
            X_tr, y_tr, y_tr_p, y_ind = sess.run([X_train, y_train, y_train_p, y_train_ind])
            X_tr= np.reshape(X_tr, (-1, 4501, 1))
            feed = {inputs_ : X_tr, labels_1 : y_tr, labels_2 : y_tr_p, keep_prob_ : 0.5, learning_rate_ : learning_rate}
            loss, _ , logit = sess.run([cost, optimizer, logits_1], feed_dict = feed)            
            y_lab = np.empty([batch_size_, 3])
            y_logit = np.empty([batch_size_, 3])
            for i in range(batch_size_):
                if y_ind[i,0] == 2:
                    y_lab[i]=np.argsort(y_tr[i])[-3:]
                    y_lab[i]=np.sort(y_lab[i])
                    y_logit[i]=np.argsort(logit[i])[-3:]
                    y_logit[i]=np.sort(y_logit[i])
                elif y_ind[i,0] == 1:
                    z=np.argsort(y_tr[i])[-2:]
                    y_lab[i]=np.append(z, [0])
                    y_lab[i]=np.sort(y_lab[i])
                    z_=np.argsort(logit[i])[-2:]
                    y_logit[i]=np.append(z_, [0])
                    y_logit[i]=np.sort(y_logit[i])
                elif y_ind[i,0] == 0:
                    z=np.argsort(y_tr[i])[-1:]
                    y_lab[i]=np.append(z, [0,0])
                    y_lab[i]=np.sort(y_lab[i])
                    z_=np.argsort(logit[i])[-1:]
                    y_logit[i]=np.append(z_, [0,0])
                    y_logit[i]=np.sort(y_logit[i])
                else:
                    print('Something Wrong happened!!!')       
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
                X_vd, y_vd, y_vd_p, y_ind_vd = sess.run([X_vld, y_vld, y_vld_p, y_vld_ind])
                X_vd= np.reshape(X_vd, (-1, 4501, 1))
                feed = {inputs_ : X_vd, labels_1 : y_vd, labels_2 : y_vd_p, keep_prob_ : 1.0, learning_rate_ : learning_rate}
                loss_vd,  logit_vd = sess.run([cost,  logits_1], feed_dict = feed)            
                y_lab_vd = np.empty([batch_size_, 3])
                y_logit_vd = np.empty([batch_size_, 3])
                for i in range(batch_size_):
                    if y_ind_vd[i,0] == 2:
                        y_lab_vd[i]=np.argsort(y_vd[i])[-3:]
                        y_lab_vd[i]=np.sort(y_lab_vd[i])
                        y_logit_vd[i]=np.argsort(logit_vd[i])[-3:]
                        y_logit_vd[i]=np.sort(y_logit_vd[i])
                    elif y_ind_vd[i,0] == 1:
                        z=np.argsort(y_vd[i])[-2:]
                        y_lab_vd[i]=np.append(z, [0])
                        y_lab_vd[i]=np.sort(y_lab_vd[i])
                        z_=np.argsort(logit_vd[i])[-2:]
                        y_logit_vd[i]=np.append(z_, [0])
                        y_logit_vd[i]=np.sort(y_logit_vd[i])
                    elif y_ind_vd[i,0] == 0:
                        z=np.argsort(y_vd[i])[-1:]
                        y_lab_vd[i]=np.append(z, [0,0])
                        y_lab_vd[i]=np.sort(y_lab_vd[i])
                        z_=np.argsort(logit_vd[i])[-1:]
                        y_logit_vd[i]=np.append(z_, [0,0])
                        y_logit_vd[i]=np.sort(y_logit_vd[i])
                    else:
                        print('Something Wrong happened!!!')
                feed_1 = {logit_num : y_logit_vd, label_num: y_lab_vd}
                acc_vd = sess.run(accuracy, feed_dict = feed_1)
                validation_acc.append(acc_vd)
                validation_loss.append(loss_vd)
                print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {:d}".format(iteration),
                        "Validation loss: {:6f}".format(loss_vd),
                        "Validation acc: {:.6f}".format(acc_vd))
            iteration += 1 
        saver.save(sess,'file path')
coord.request_stop()
coord.join(threads)
    

