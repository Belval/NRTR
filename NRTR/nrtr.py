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
                self.__targets,
                self.__init,
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

            concat1 = tf.reshape(bnorm2, (bnorm2.get_shape().as_list()[0:2] + [512,]))

            # batch_size x 25 x 512

            return concat1
        
        def multi_head_attention(q, k, v):
            """
                Multi-head Attention as described in paper (p.3)
                    In the "Exploration of the core module architectures" part, the head count is set to 8
                    This is coherent with the "All You Need Is Attention" paper (https://arxiv.org/pdf/1706.03762v5.pdf)
            """

            def scaled_dot_product_attention(q, k, v, masked=False):
                """
                    Scaled dot-product Attention as described in paper (p.3)
                """

                # We start by doing the queries with the transposed of the keys 
                qk = tf.matmul(q, k, transpose_b=True)

                if masked:
                    raise NotImplementedError("Masked scaled dot product is not implemented")

                # We then softmax the result divided by the sqrt of the width of the keys
                sm1 = tf.nn.softmax(tf.divide(qk, k.get_shape().as_list()[-1]**0.5))

                return tf.matmul(sm1, v)

            def linear_projection(q, k, v):
                """
                    Linear projection of the queries, keys, and values
                """
            
                ql_1 = tf.layers.dense(q, q.get_shape().as_list()[2])
                kl_1 = tf.layers.dense(k, k.get_shape().as_list()[2])
                vl_1 = tf.layers.dense(v, v.get_shape().as_list()[2])

                return ql_1, kl_1, vl_1

            def split_heads(q, k, v):
                """
                    Split the heads, partially taken from https://github.com/DongjunLee/transformer-tensorflow/blob/b6585fa7504f0f35327f2a3994dac7b06b6036f7/transformer/attention.py#L57
                """

                def split_last_dimension_then_transpose(tensor, num_heads, dim):
                    t_shape = tensor.get_shape().as_list()
                    tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
                    return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

                qs = split_last_dimension_then_transpose(q, 8, q.get_shape().as_list()[2])
                ks = split_last_dimension_then_transpose(k, 8, k.get_shape().as_list()[2])
                vs = split_last_dimension_then_transpose(v, 8, v.get_shape().as_list()[2])

                return qs, ks, vs

            def concat_heads(heads):
                """
                    Concatenate the result of the scaled dot-product attention
                """

                heads_t = tf.transpose(heads, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
                heads_shape = heads_t.get_shape().as_list()
                num_heads, dim = heads_shape[-2:]
                return tf.reshape(heads_t, [-1] + heads_shape[1:-2] + [num_heads * dim])

            # So all the building blocks exists, we only have to assemble them together
            ql, kl, vl = linear_projection(q, k, v)
            qs, ks, vs = split_heads(ql, kl, vl)
            sdp_1 = scaled_dot_product_attention(qs, ks, vs)
            concat_1 = concat_heads(sdp_1)
            linear_1 = tf.layers.dense(concat_1, concat_1.get_shape().as_list()[-1])

            return linear_1

        def position_wise_feed_forward_network(x):
            """
                Position-wise Feed-Forward Network as described in paper (p.4)
            """

            # First linear
            linear_1 = tf.layers.dense(x, x.get_shape().as_list()[-1])

            # ReLU operation
            relu_1 = tf.nn.relu(linear_1)

            # Second linear
            linear_2 = tf.layers.dense(relu_1, relu_1.get_shape().as_list()[-1])

            return linear_2

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

        def encoder(x):
            """
                Encoder structure as described in paper (p.4)
            """

            # First modality transform block
            mtb_1 = modality_transform_block(x)

            # Linear
            linear_1 = tf.layers.dense(mtb_1, mtb_1.get_shape().as_list()[2])

            # Positional Encoding
            positional_encoding_1 = positional_encoding(linear_1)

            # Multi-Head Attention
            mha_1 = multi_head_attention(positional_encoding_1, positional_encoding_1, positional_encoding_1)

            # Layer norm 1
            ln_1 = layer_norm(mha_1)

            # Add op with previous
            add_1 = tf.add(ln_1, positional_encoding_1)

            # FFN
            ffn_1 = position_wise_feed_forward_network(add_1)
            
            # Layer norm 2
            ln_2 = layer_norm(ffn_1)

            # Add op with previous
            add_2 = tf.add(ln_2, add_1)

            return add_2

        def decoder(x):
            """
                Decoder structure as described in paper (p.4)
            """

            return x

        inputs = tf.placeholder(tf.float32, [batch_size, max_width, 32, 1])
        targets = tf.placeholder(tf.float32, [batch_size, None, config.NUM_CLASSES])

        # Here 6 is arbitrary, according to paper is could be anywhere between 4 and 12
        for _ in range(6):
            encoder_outputs = encoder(inputs)

        encoder_outputs = layer_norm(encoder_outputs)

        # Here 6 is arbitrary, according to paper is could be anywhere between 4 and 12
        for _ in range(6):
            decoder_outputs = decoder(encoder_outputs)

        decoder_outputs = layer_norm(decoder_outputs)

        output_probabilities = tf.layers.dense(decoder_outputs, config.NUM_CLASSES, activation=tf.contrib.layers.softmax)

        # Original paper does not mention a loss or cost function
        loss = tf.losses.softmax_cross_entropy(targets, output_probabilities)

        # Learning rate is defined by a formula in the original paper. The 0.001 value is a placeholder
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(loss)

        init = tf.global_variables_initializer()

        return inputs, targets, init

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
