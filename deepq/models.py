import tensorflow as tf
import tensorflow.contrib.layers as layers

def _cnn_to_mlp(convs, hiddens, num_actions, inpt, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            distribution_list = []
            num_atoms = 51
            for i in range(num_actions):
                distribution_list.append(tf.layers.dense(inputs=action_out, units = num_atoms,activation='softmax'))
            q_out = distribution_list
            return q_out

def cnn_to_mlp(convs, hiddens, num_actions, layer_norm=False):
    return lambda *args, **kwargs: _cnn_to_mlp(convs, hiddens, num_actions=num_actions,
                                               layer_norm=layer_norm, *args, **kwargs)
