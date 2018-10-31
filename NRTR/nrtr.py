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
            (
                self.__inputs,
                self.__seq_len,
                self.__targets_1,
                self.__targets_2,
                self.__iteration_n,
                self.__is_training,
                self.__output,
                self.__loss,
                self.__learning_rate,
                self.__optimizer,
                self.__init,
                self.__a1,
                self.__a2,
                self.__weight_mask
            ) = self.nrtr(max_image_width, batch_size)

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
        self.__data_manager = DataManager(batch_size, model_path, examples_path, max_image_width, train_test_ratio)

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

            concat1 = tf.reshape(bnorm2, (bnorm2.get_shape().as_list()[0:2] + [512,]))

            # batch_size x 25 x 512

            return concat1
        
        def multi_head_attention(q, k, v, is_training, masked=False):
            """
                Multi-head Attention as described in paper (p.3)
                    In the "Exploration of the core module architectures" part, the head count is set to 8
                    This is coherent with the "All You Need Is Attention" paper (https://arxiv.org/pdf/1706.03762v5.pdf)
            """

            def scaled_dot_product_attention(q, k, v, qs, ks, vs, masked=False):
                """
                    Scaled dot-product Attention as described in paper (p.3)
                """

                # We start by doing the queries with the transposed of the keys 
                qk = tf.matmul(qs, tf.transpose(ks, [0, 2, 1]))

                div_qk = tf.divide(qk, 64**0.5)

                # Key Masking
                key_masks = tf.sign(tf.abs(tf.reduce_sum(k, axis=-1))) # (N, T_k)
                key_masks = tf.tile(key_masks, [8, 1]) # (h*N, T_k)
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(q)[1], 1]) # (h*N, T_q, T_k)
                paddings = tf.ones_like(div_qk)*(-2**32+1)
                div_qk = tf.where(tf.equal(key_masks, 0), paddings, div_qk) # (h*N, T_q, T_k)

                if masked:
                    diag_vals = tf.ones_like(div_qk[0, :, :]) # (T_q, T_k)
                    tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
                    masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(div_qk)[0], 1, 1]) # (h*N, T_q, T_k)
                    paddings = tf.ones_like(masks)*(-2**32+1)
                    div_qk = tf.where(tf.equal(masks, 0), paddings, div_qk) # (h*N, T_q, T_k)

                # We then softmax the result divided by the sqrt of the width of the keys
                sm1 = tf.nn.softmax(div_qk)

                # Query masking
                query_masks = tf.sign(tf.abs(tf.reduce_sum(q, axis=-1))) # (N, T_q)
                query_masks = tf.tile(query_masks, [8, 1]) # (h*N, T_q)
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(k)[1]]) # (h*N, T_q, T_k)
                sm1 *= query_masks

                return tf.matmul(sm1, vs)

            def linear_projection(q, k, v):
                """
                    Linear projection of the queries, keys, and values
                """
            
                ql_1 = tf.layers.dense(q, q.get_shape().as_list()[2], activation=tf.nn.relu)
                kl_1 = tf.layers.dense(k, k.get_shape().as_list()[2], activation=tf.nn.relu)
                vl_1 = tf.layers.dense(v, v.get_shape().as_list()[2], activation=tf.nn.relu)

                return ql_1, kl_1, vl_1

            def split_heads(q, k, v):
                """
                    Split the heads, partially taken from https://github.com/DongjunLee/transformer-tensorflow/blob/b6585fa7504f0f35327f2a3994dac7b06b6036f7/transformer/attention.py#L57
                """

                qs = tf.concat(tf.split(q, 8, axis=2), axis=0)
                ks = tf.concat(tf.split(k, 8, axis=2), axis=0)
                vs = tf.concat(tf.split(v, 8, axis=2), axis=0)

                return qs, ks, vs

            def concat_heads(heads):
                """
                    Concatenate the result of the scaled dot-product attention
                """

                return tf.concat(tf.split(heads, 8, axis=0), axis=2)

            # So all the building blocks exists, we only have to assemble them together
            ql, kl, vl = linear_projection(q, k, v)
            qs, ks, vs = split_heads(ql, kl, vl)
            sdp_1 = scaled_dot_product_attention(q, k, v, qs, ks, vs, masked)
            concat_1 = concat_heads(sdp_1)
            linear_1 = tf.layers.dense(concat_1, concat_1.get_shape().as_list()[-1])

            return tf.layers.dropout(linear_1, 0.1, is_training)

        def position_wise_feed_forward_network(x, is_training):
            """
                Position-wise Feed-Forward Network as described in paper (p.4)
            """

            # First linear
            linear_1 = tf.layers.dense(x, x.get_shape().as_list()[-1])

            # ReLU operation
            relu_1 = tf.nn.relu(linear_1)

            # Second linear
            linear_2 = tf.layers.dense(relu_1, relu_1.get_shape().as_list()[-1])

            return tf.layers.dropout(linear_2, 0.1, is_training)

        def layer_norm(x):
            """
                Layer normalization as described in paper (p.4)
            """

            # I'm using the TensorFlow version, I have no reason to think that the original paper did the same thing
            return tf.contrib.layers.layer_norm(x)

        def positional_encoding(x):
            """
                Not as described in paper since it lacked proper description of this step.
                This function is based on the "Attention is all you need" paper.
            """

            seq_len, dim = x.get_shape().as_list()[-2:]
            encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(seq_len) for i in range(dim)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])
            encoded_vec_tensor = tf.convert_to_tensor(encoded_vec.reshape([seq_len, dim]), dtype=tf.float32)
            return tf.add(x, encoded_vec_tensor)

        def encoder(x, is_training):
            """
                Encoder structure as described in paper (p.4)
            """

            # Multi-Head Attention
            mha_1 = multi_head_attention(x, x, x, is_training)

            # Layer norm 1
            ln_1 = layer_norm(mha_1)

            # Add op with previous
            add_1 = tf.add(ln_1, x)

            # FFN
            ffn_1 = position_wise_feed_forward_network(add_1, is_training)
            
            # Layer norm 2
            ln_2 = layer_norm(ffn_1)

            # Add op with previous
            add_2 = tf.add(ln_2, add_1)

            return add_2

        def decoder(x, positional_encoding, is_training):
            """
                Decoder structure as described in paper (p.4)
            """

            # Multi-Head Attention
            mha_1 = multi_head_attention(positional_encoding, positional_encoding, positional_encoding, is_training, masked=True)

            # Layer norm 1
            ln_1 = layer_norm(mha_1)

            # Add op with previous
            add_1 = tf.add(ln_1, positional_encoding)

            # Multi-Head Attention
            mha_2 = multi_head_attention(x, x, positional_encoding, is_training)

            # Layer norm 2
            ln_2 = layer_norm(mha_2)

            # Add op with previous
            add_2 = tf.add(ln_2, add_1)

            # FFN
            ffn_1 = position_wise_feed_forward_network(add_2, is_training)

            # Layer norm 3
            ln_3 = layer_norm(ffn_1)

            # Add op with previous
            add_3 = tf.add(ln_3, add_2)

            return add_3

        inputs = tf.placeholder(tf.float32, [batch_size, max_width, 32, 1])

        is_training = tf.placeholder(tf.bool, name='is_training')

        seq_len = tf.placeholder(tf.int32, [batch_size])
        targets_1 = tf.placeholder(tf.float32, [batch_size, 25, config.NUM_CLASSES])
        targets_2 = tf.placeholder(tf.int32, [batch_size, 25])

        # The iteration number, used in calculating the learning rate
        iteration_n = tf.placeholder(tf.float32, [1])
        # Define if we are training, used by dropout layers

        # First modality transform block
        mtb_1 = modality_transform_block(inputs)

        # Linear
        linear_1 = tf.layers.dense(mtb_1, mtb_1.get_shape().as_list()[2])

        # Positional Encoding
        positional_encoding_1 = positional_encoding(linear_1)

        # Obviously it's not a real encoder output yet
        encoder_outputs = positional_encoding_1

        # Here 6 is arbitrary, according to paper it could be anywhere between 4 and 12
        for _ in range(6):
            encoder_outputs = encoder(encoder_outputs, is_training)

        encoder_outputs = layer_norm(encoder_outputs)

        # Positional encoding
        positional_encoding_2 = positional_encoding(targets_1)

        # Here 6 is arbitrary, according to paper it could be anywhere between 4 and 12
        for i in range(6):
            decoder_outputs = decoder(encoder_outputs if i == 0 else decoder_outputs, positional_encoding_2, is_training)

        decoder_outputs = layer_norm(decoder_outputs)

        output_probabilities = tf.layers.dense(decoder_outputs, config.NUM_CLASSES, activation=tf.contrib.layers.softmax)

        weight_masks = tf.sequence_mask(lengths=seq_len, maxlen=25, dtype=tf.float32)

        # Original paper does not mention a loss or cost function
        loss = tf.contrib.seq2seq.sequence_loss(logits=output_probabilities, targets=targets_2, weights=weight_masks)

        # I did not implement pretraining so we'll use only the relevant part of the equation presented p.5
        learning_rate = tf.reduce_sum(tf.pow(tf.cast(mtb_1.get_shape().as_list()[2], tf.float32), -0.5) * tf.pow(iteration_n, -0.5))

        # Learning rate is defined by a formula in the original paper. The 0.001 value is a placeholder
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(loss)

        init = tf.global_variables_initializer()

        return inputs, seq_len, targets_1, targets_2, iteration_n, is_training, output_probabilities, loss, learning_rate, optimizer, init, encoder_outputs, decoder_outputs, weight_masks

    def train(self, iteration_count):
        with self.__session.as_default():
            print('Training')
            for i in range(self.step, iteration_count + self.step):
                iter_loss = 0
                for batch_y, batch_seq_len, batch_dt, batch_dt_2, batch_x in self.__data_manager.train_batches:
                    _, output, loss_value, learning_rate, a1, a2, wm = self.__session.run(
                        [self.__optimizer, self.__output, self.__loss, self.__learning_rate, self.__a1, self.__a2, self.__weight_mask],
                        feed_dict={
                            self.__inputs: batch_x,
                            self.__seq_len: batch_seq_len,
                            self.__targets_1: batch_dt,
                            self.__targets_2: batch_dt_2,
                            self.__iteration_n: [float(i) + 1.], # +1 because 0^(0.5) is undefined obviously
                            self.__is_training: True
                        }
                    )

                    #print(batch_y[0])
                    #print(wm[0])
                    #input()
#
                    if i % 10 == 0:
                        for j in range(2):
                            print(batch_y[j])
                            print(batch_dt[j])
                            print(ground_truth_to_word(output[j]))
                    #    print(output[0][0][0:15])
                    #    print('------------')
                    #    print(a2[0][0][0:15])
                    #    print('------------')
                    #    print(a2[1][0][0:15])
                    #    input()

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
            for batch_y, batch_dt, batch_x in self.__data_manager.test_batches:
                print("blah")
                output = self.__session.run(
                    self.__output,
                    feed_dict={
                        self.__inputs: batch_x,
                        self.__targets_1: batch_dt,
                        self.__is_training: False
                    }
                )

                for i, y in enumerate(batch_y):
                    print(batch_y[i])
                    print(batch_x[i])
                    print(ground_truth_to_word(output[i]))
