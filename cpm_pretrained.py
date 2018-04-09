import tensorflow as tf


class CpmModel(object):
    def __init__(self):
        self.saver = tf.train.import_meta_graph('model/pose_net_v2.ckpt.meta')
        self.graph = tf.get_default_graph()

    def restore(self, sess):
        self.saver.restore(sess, 'model/pose_net_v2.ckpt')

    def get_input_placeholder(self):
        return self.graph.get_tensor_by_name('CPM/Placeholder_1:0')

    def get_output_conv4_5(self):
        conv4_5 = self.graph.get_tensor_by_name('CPM/PoseNet/conv4_5_CPM/BiasAdd:0')
        return tf.stop_gradient(conv4_5)