import os
import time
import numpy as np
import tensorflow as tf
import config
from scipy.misc import imread, imresize, imsave

from data_manager import DataManager
from utils import resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NRTR(object):
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio, restore):
        self.step = 0
        self.__model_path = model_path
        self.__save_path = os.path.join(model_path, 'ckp')

        self.__restore = restore

        self.__training_name = str(int(time.time()))
        self.__session = tf.Session()

        # Building graph
        with self.__session.as_default():
            TODO = self.nrtr(max_image_width, batch_size)
            self.__init.run()

        with self.__session.as_default():
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            # Loading last save if needed
            if self.__restore:
                print('Restoring')
                ckpt = tf.train.latest_checkpoint(self.__model_path)
                if ckpt:
                    print('Checkpoint is valid')
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)

        # Creating data_manager
        self.__data_manager = DataManager(batch_size, model_path, examples_path, max_image_width, train_test_ratio, self.__max_char_count)

    def nrtr(self, max_width, batch_size):
        """
            Builds the graph, returns the "important" ops
        """

        def modality_transform_block(inputs):
            """
                CNN-Only Modality tranform block as described in paper (p.6)
            """

            # batch_size x 100 x 32
            
            conv1 = tf.layers.conv2d(inputs=inputs, filters=32, kernel_size=(3, 3), strides=2, padding="same")

            relu1 = tf.nn.relu(conv1)

            bnorm1 = tf.layers.batch_normalization(relu1)

            # batch_size x 50 x 16

            conv2 = tf.layers.conv2d(inputs=bnorm1, filters=64, kernel_size=(3, 3), strides=2, padding="same")

            relu2 = tf.nn.relu(conv2)

            bnorm2 = tf.layers.batch_normalization(relu2)

            # batch_size x 25 x 8 x 64 

            concat1 = tf.concat(bnorm2, 2)

            # batch_size x 25 x 512

            return concat1
        
        def multi_head_attention(q, k, v):
            """
                Multi-head Attention as described in paper (p.3)
            """

            def scaled_dot_product_attention(q, k, v):
                """
                    Scaled dot-product Attention as described in paper (p.3)
                """

                # We start by doing the queries with the transposed of the keys 
                qk = tf.matmul(q, tf.transpose(k))

                # We then softmax the result divided by the sqrt of the width of the keys
                sm1 = tf.softmax(tf.divide(qk, tf.sqrt(tf.shape(k)[1])))

                return tf.matmul(sm1, v)

            

        return None

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            for i in range(self.step, iteration_count + self.step):
                iter_loss = 0
                for batch_y, batch_dt, batch_x in self.__data_manager.train_batches:
                    op, decoded, loss_value = self.__session.run(
                        [self.__optimizer, self.__decoded, self.__cost],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__targets: batch_dt
                        }
                    )

                    if i % 10 == 0:
                        for j in range(2):
                            print(batch_y[j])
                            print(ground_truth_to_word(decoded[j]))

                    iter_loss += loss_value

                self.__saver.save(
                    self.__session,
                    self.__save_path,
                    global_step=self.step
                )

                print('[{}] Iteration loss: {}'.format(self.step, iter_loss))

                self.step += 1
        return None

    def test(self):
        with self.__session.as_default():
            print('Testing')
            for batch_y, _, batch_x in self.__data_manager.test_batches:
                decoded = self.__session.run(
                    self.__decoded,
                    feed_dict={
                        self.__inputs: batch_x,
                    }
                )

                for i, y in enumerate(batch_y):
                    print(batch_y[i])
                    print(ground_truth_to_word(decoded[i]))
return None