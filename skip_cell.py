'''Author: Brandon Trabucco, Copyright 2019
Implements a novel skip connection gating mechanism that enables 
sequence to sequence learning.'''


import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.python.keras.utils import tf_utils
import collections


def _softmax_attention(x):
    # x is shaped: [batch, num_boxes, depth]
    x = tf.transpose(x, [0, 2, 1])
    return tf.transpose(tf.nn.softmax(x), [0, 2, 1])


def _sigmoid_attention(x):
    # x is shaped: [batch, num_boxes, depth]
    x_size = tf.to_float(tf.shape(x)[1])
    return tf.nn.sigmoid(x) / x_size


def _make_tuple_class(n):
     # Used to store the internal states of each LSTM.
    _SkipStateTuple = collections.namedtuple(
        "SkipStateTuple", ['n' + str(i) for i in range(n)])
    # Wrapper for _SkipStateTuple.
    class SkipStateTuple(_SkipStateTuple):
        """Tuple used by SkipCells for `state_size`, `zero_state`, and output state.
        Stores n elements: `(0, 1, ... n - 1)`, in that order.
        """
        __slots__ = ()
        @property
        def dtype(self):
            contents = self
            v = contents[0]
            for l in contents[1:]:
                if not v.dtype == l.dtype:
                    raise TypeError("Inconsistent internal state: %s vs %s" % (
                        str(v.dtype), str(l.dtype)))
            return v.dtype
    return SkipStateTuple


# The wrapper for the up-down attention mechanism
class SkipCell(tf.contrib.rnn.LayerRNNCell):
    '''Implements a novel skip connection gating mechanism that enables 
    sequence to sequence learning.'''

    def __init__(self, 
            # The default Dense layer parameters
            units, activation=None, use_bias=True,
            kernel_initializer=None, bias_initializer=None,
            kernel_regularizer=None, bias_regularizer=None,
            activity_regularizer=None, kernel_constraint=None,
            bias_constraint=None, trainable=True, name=None, 
            # The extra parameters for the skip mechanism
            reuse=None, skip_depth=2, skip_method='last', 
            attention_method='softmax', **kwargs):

        '''Builds a skip layer of Densely-connected layers.
        Args:
            units: Integer or Long, dimensionality of the output space.
            activation: Activation function (callable). Set it to None to maintain a
                linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            kernel_initializer: Initializer function for the weight matrix.
                If `None` (default), weights are initialized using the default
                initializer used by `tf.get_variable`.
            bias_initializer: Initializer function for the bias.
            kernel_regularizer: Regularizer function for the weight matrix.
            bias_regularizer: Regularizer function for the bias.
            activity_regularizer: Regularizer function for the output.
            kernel_constraint: An optional projection function to be applied to the
                kernel after being updated by an `Optimizer` (e.g. used to implement
                norm constraints or value constraints for layer weights). The function
                must take as input the unprojected variable and must return the
                projected variable (which must have the same shape). Constraints are
                not safe to use when doing asynchronous distributed training.
            bias_constraint: An optional projection function to be applied to the
                bias after being updated by an `Optimizer`.
            trainable: Boolean, if `True` also add variables to the graph collection
                `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
            name: String, the name of the layer. Layers with the same name will
                share weights, but to avoid mistakes we require reuse=True in such cases.
            reuse: Boolean, whether to reuse the weights of a previous layer
                by the same name.
            skip_depth: int, the amount of inner layers to build skip connections between. 
                (optional)
            skip_method: string, the method to calculate the return output of the skip 
                mechanism, must be from ['last', 'mean', 'max', 'attention'] (optional)
            attention_method: string, either 'softmax' or 'sigmoid' (optional)
            **kwargs: Dict, keyword named properties for common layer attributes, like
                `trainable` etc when constructing the cell from configs of get_config().
        '''
        
        super(SkipCell, self).__init__(
            _reuse=reuse, name=name, **kwargs)

        if skip_method not in ['last', 'mean', 'max', 'attention']:
            raise Exception('skip_method must be in %s.' % str(
                ['last', 'mean', 'max', 'attention']))
        if attention_method not in ['softmax', 'sigmoid']:
            raise Exception('attention_method must be in %s.' % str(
                ['softmax', 'sigmoid']))
        self.skip_method = skip_method
        self.attn_fn = {'sigmoid': _sigmoid_attention, 
            'softmax': _softmax_attention}[attention_method]

        # Create the dense layers connected with skips
        self.skip_layers = [tf.layers.Dense(
            units, activation=activation, use_bias=use_bias,
            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, 
            trainable=trainable, name=name) for _ in range(skip_depth)]

        self.attn_layer = None
        if skip_method == 'attention':
            self.attn_layer = tf.layers.Conv1D(1, 3, kernel_initializer=kernel_initializer, 
                padding="same", activation=self.attn_fn, name="attention")

        # Create a variably sized state tuple class
        self.SkipStateTuple = _make_tuple_class(skip_depth)
                
        # For managing the RNN functions such as 'zero_state'
        self._state_size = self.SkipStateTuple(*[units for _ in range(skip_depth)])
        self._output_size = units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state):
        
        # Each skip layer only sees what is behind it
        next_state = [l(tf.concat([inputs] + list(state[:i]), 1)) for i, l in enumerate(
            self.skip_layers)]
        if self.skip_method == 'last':
            outputs = next_state[-1]
        elif self.skip_method == 'mean':
            outputs = tf.reduce_mean(tf.stack(next_state, 1), 1)
        elif self.skip_method == 'max':
            outputs = tf.reduce_max(tf.stack(next_state, 1), 1)
        elif self.skip_method == 'attention':
            all_states = tf.stack(next_state, 1)
            outputs = tf.reduce_sum(all_states * self.attn_layer(all_states), 1)
        return outputs, self.SkipStateTuple(*next_state)