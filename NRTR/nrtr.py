import os
import time
import numpy as np
import tensorflow as tf
from tf.keras import layers
import config
from scipy.misc import imread, imresize, imsave

from .data_manager import DataManager
from .utils import resize_image, label_to_array, ground_truth_to_word, levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class NRTR:
    def __init__(self, batch_size, model_path, examples_path, max_image_width, train_test_ratio, restore):
        self.step = 0
        self.batch_size
        self.model_path = model_path
        self.max_image_width = max_image_width
        self.save_path = os.path.join(model_path, 'ckp')
        self.restore = restore
        self.training_name = str(int(time.time()))
        self.is_training = False

        # Creating data_manager
        self.data_manager = DataManager(batch_size, model_path, examples_path, max_image_width, train_test_ratio)

    def build(self, inputs, max_width, batch_size):
        """
            Builds the graph, returns the "important" ops
        """

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
        linear_1 = layers.Dense(mtb_1.get_shape().as_list()[2])(mtb_1)

        # Positional Encoding
        positional_encoding_1 = positional_encoding(linear_1)

        # Obviously it's not a real encoder output yet
        encoder_outputs = positional_encoding_1

        # Here 6 is arbitrary, according to paper it could be anywhere between 4 and 12
        for _ in range(6):
            encoder_outputs = encoder(encoder_outputs, self.__is_training)

        encoder_outputs = layer_norm(encoder_outputs)

        # Positional encoding
        positional_encoding_2 = positional_encoding(targets_1)

        # Here 6 is arbitrary, according to paper it could be anywhere between 4 and 12
        for i in range(6):
            decoder_outputs = decoder(encoder_outputs if i == 0 else decoder_outputs, positional_encoding_2, self.__is_training)

        decoder_outputs = layer_norm(decoder_outputs)

        output_probabilities = layers.Dense(config.NUM_CLASSES, activation=tf.contrib.layers.softmax)(decoder_outputs)

        weight_masks = tf.sequence_mask(lengths=seq_len, maxlen=25, dtype=tf.float32)

        # Original paper does not mention a loss or cost function
        loss = tf.contrib.seq2seq.sequence_loss(logits=output_probabilities, targets=targets_2, weights=weight_masks)

        # I did not implement pretraining so we'll use only the relevant part of the equation presented p.5
        learning_rate = tf.reduce_sum(tf.pow(tf.cast(mtb_1.get_shape().as_list()[2], tf.float32), -0.5) * tf.pow(iteration_n, -0.5))

        # Learning rate is defined by a formula in the original paper. The 0.001 value is a placeholder
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(loss)

        init = tf.global_variables_initializer()

        return inputs, seq_len, targets_1, targets_2, iteration_n, output_probabilities, loss, learning_rate, optimizer, init, encoder_outputs, decoder_outputs, weight_masks

    def modality_transform_block(inputs):
        """
            CNN-Only Modality tranform block as described in paper (p.6)
        """

        # batch_size x 100 x 32
        
        conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same")(inputs)

        relu1 = tf.nn.relu(conv1)

        bnorm1 = layers.BatchNormalization()(relu1)

        # batch_size x 50 x 16

        conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same")(bnorm1)

        relu2 = tf.nn.relu(conv2)

        bnorm2 = layers.BatchNormalization()(relu2)

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
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
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
        
            ql_1 = layers.Dense(q.get_shape().as_list()[2], activation=tf.nn.relu)(q)
            kl_1 = layers.Dense(k.get_shape().as_list()[2], activation=tf.nn.relu)(k)
            vl_1 = layers.Dense(v.get_shape().as_list()[2], activation=tf.nn.relu)(v)

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
        linear_1 = layers.Dense(concat_1.get_shape().as_list()[-1])(concat_1)

        if self.__is_training:
            return layers.Dropout(0.1)(linear_1)
        else:
            return linear_1

    def position_wise_feed_forward_network(x, is_training):
        """
            Position-wise Feed-Forward Network as described in paper (p.4)
        """

        # First linear
        linear_1 = layers.Dense(x.get_shape().as_list()[-1])(x)

        # ReLU operation
        relu_1 = tf.nn.relu(linear_1)

        # Second linear
        linear_2 = layers.Dense(relu_1.get_shape().as_list()[-1])(relu_1)

        if self.__is_training:
            return layers.Dropout(0.1)(linear_2)
        else:
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
