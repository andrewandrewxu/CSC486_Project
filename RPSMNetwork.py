import os
import tensorflow as tf
import numpy as np
from pretrained.Pose2dFeatureExtraction import Pose2d

"Original base for the code from CSC486B/CSC586B by Kwang Moo Yi, released under the MIT license"

class RPSMNetwork(object):
    """Network class """

    def __init__(self, config):

        self.config = config
        self.num_states = 1024
        self.image_size = 368
        self.num_frames = 16
        self.num_output = 51

        self._build_placeholder()
        self._init_recurrent()
        self._build_2d_pose_module()
        self._build_feature_adaption_module()
        self._build_3d_pose_recurrent_module()
        self._build_loss()
        self._build_optim()
        self._build_summary()
        self._build_writer()

    def _init_recurrent(self):
        with tf.variable_scope("Recurrent", reuse=tf.AUTO_REUSE):
            self.lstm = tf.contrib.rnn.BasicLSTMCell(1024)
            hidden_state = tf.get_variable('hidden_state', dtype=tf.float32, shape=(1,1024),
                                           initializer=tf.initializers.zeros)
            current_state = tf.get_variable('current_state', dtype=tf.float32, shape=(1, 1024),
                                            initializer=tf.initializers.zeros)
            self.state = tf.contrib.rnn.LSTMStateTuple(hidden_state,  current_state)

            # 2d pose module 46x46x128 output from prev stage
            self.in_2d_pose = tf.get_variable('in_2d_pose', dtype=tf.float32, shape=(46, 46, 128),
                                              initializer=tf.initializers.zeros, trainable=False)
            # output from 3D Pose Recurrent Module from prev stage

            self.in_3d_pose_fc = tf.get_variable('in_3d_pose_fc', dtype=tf.float32, shape=(1, self.num_output),
                                                 initializer=tf.initializers.zeros, trainable=False)

    def _build_placeholder(self):
        """Build placeholders."""
        self.x_in = tf.placeholder(tf.float32, shape=(None, self.image_size, self.image_size, 3))
        self.features_in = tf.placeholder(tf.float32, shape=(46, 46, 128))
        self.y_in = tf.placeholder(dtype=tf.float32, shape=self.num_output)

    def _build_2d_pose_module(self):
        with tf.variable_scope("2dPoseModule", reuse=tf.AUTO_REUSE):
            concat = tf.concat([self.features_in, self.in_2d_pose], axis=2)
            conv1 = tf.layers.conv2d(tf.reshape(concat, shape=(1, 46, 46, 256)), 128, 3, 1, padding='SAME', activation=tf.nn.relu)
            self.out_2d_pose = tf.layers.conv2d(conv1, 128, 3, 1, padding='SAME', activation=tf.nn.relu)
            self.output_2d_assign = tf.assign(self.in_2d_pose, tf.reshape(self.out_2d_pose, shape=(46, 46, 128)))

    def _build_feature_adaption_module(self):
        with tf.variable_scope("FeatureAdaptionModule", reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(self.out_2d_pose, 128, 5, 2)
            conv2 = tf.layers.conv2d(conv1, 128, 5, 2)
            max = tf.layers.max_pooling2d(conv2, 2, 2)
            flat = tf.layers.flatten(max)
            self.feature_adaption_output = tf.layers.dense(flat, self.num_states)

    def _build_3d_pose_recurrent_module(self):
        with tf.variable_scope("3dPoseRecurrentModule", reuse=tf.AUTO_REUSE):
            # TODO figure out how hidden states work and the dimensions
            concat = tf.concat([self.in_3d_pose_fc, self.feature_adaption_output], axis=-1)
            output, state = self.lstm(concat, self.state)
            self.state_assign = tf.group(tf.assign(self.state.c,  state.c), tf.assign(self.state.h, state.h))
            self.pred = tf.layers.dense(output, 51)
            self.out_pred_assign = tf.assign(self.in_3d_pose_fc, self.pred)

    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            # Euclidean distances between the prediction for all P joints and ground truth:
            joints_pred = tf.reshape(self.pred, shape=(17, 3))
            joints_truth = tf.reshape(self.y_in, shape=(17, 3))
            dist = tf.subtract(joints_pred, joints_truth)
            dist = tf.square(dist)
            sum = tf.reduce_sum(dist, axis=1)
            self.loss = tf.reduce_sum(tf.sqrt(sum))

            # Record summary for loss
            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optim", reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable(
                "global_step", shape=(),
                initializer=tf.zeros_initializer(),
                dtype=tf.int64,
                trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=1)
            # TODO get learning rate from config
            self.optim = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""
        # TODO get config vars

        # # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join('logs', "train"))
        # self.summary_va = tf.summary.FileWriter(
        #     os.path.join(self.config.log_dir, "valid"))
        # # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # # Save file for the current model
        # self.save_file_cur = os.path.join(
        #     self.config.log_dir, "model")
        # # Save file for the best model
        # self.save_file_best = os.path.join(
        #     self.config.save_dir, "model")

    def train(self, x_tr, y_tr, x_va, y_va):
        """Training function.

        Parameters
        ----------
        x_tr : ndarray
            Training data.

        y_tr : ndarray
            Training labels.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """
        pose2d = Pose2d({'data':  self.x_in})

        # Run TensorFlow Session
        with tf.Session() as sess:
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())
            pose2d.load('pretrained/Pose2dFeatureExtraction.npy', sess)


            # b_resume = tf.train.latest_checkpoint(self.save_file_cur)
            # if b_resume:
            #     print("Restoring from {}...".format(
            #         self.config.log_dir))
            #     self.saver_cur.restore(sess, self.save_file_cur)
            #     step = tf.get_variable("step", shape=1)
            #     best_acc = tf.get_variable("best_va_acc", shape=1)
            # else:
            #     print("Starting from scratch...")
            #     step = 0
            #     best_acc = 0

            print("Training...")
            for x_seq, y_seq in zip(x_tr, y_tr):
                self._init_recurrent()
                features = sess.run(pose2d.get_output(), feed_dict={self.x_in: x_seq})
                # x_f and y_f are frames for one image sequence
                for i, (x_f, y_f) in enumerate(zip(x_seq, y_seq)):
                    res = sess.run(
                        fetches={
                            "optim": self.optim,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                            "2dout": self.output_2d_assign,  # assign 2d pose features used in next step
                            "3dout": self.out_pred_assign,  # assign 3d pose used in next step
                            "state": self.state_assign # assign state used for next step
                        },
                        feed_dict={
                            self.features_in: features[i],
                            self.y_in: y_f
                        },
                    )
                # Write Training Summary
                self.summary_tr.add_summary(
                    res["summary"], global_step=res["global_step"],
                )

                # TODO test on validation set
                # save best model
            self.saver_cur.save(sess, './logs/model.ckpt')
            self.summary_tr.add_graph(sess.graph)

    def test(self, x_te, y_te):
        """Test routine"""
        #TODO config
        with tf.Session() as sess:
            # Load the best model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.config.save_dir)
            if tf.train.latest_checkpoint(self.config.save_dir) is not None:
                print("Restoring from {}...".format(
                    self.config.save_dir))
                self.saver_best.restore(
                    sess,
                    latest_checkpoint
                )

        #Evaluate the accuracy of the tested trained data compared to model and prints results
            res = sess.run(
               fetches={
                    "acc": self.acc,
               },
               feed_dict={
                    self.x_in: x_te,
                    self.y_in: y_te,
               },
            )
            print("Test accuracy is {}".format(res["acc"]))



def main(config):
    """The main function."""

    # TODO load data

    mynet = RPSMNetwork(config)
    x_tr = np.zeros((1, 16, 368, 368, 3))
    y_tr = np.zeros((1, 16, 51))
    x_va = np.zeros((1, 16, 368, 368, 3))
    y_va = np.zeros((1, 16, 51))

    mynet.train(x_tr, y_tr, x_va, y_va)

    x_te = np.zeros((1, 16, 376, 376, 3))
    y_te = np.zeros((1, 16, 51))

    #mynet.test(x_te, y_te)


if __name__ == "__main__":
    #
    # # TODO setup config file
    # # ----------------------------------------
    # # Parse configuration
    # config, unparsed = get_config()
    # # If we have unparsed arguments, print usage and exit
    # if len(unparsed) > 0:
    #     print_usage()
    #     exit(1)

    main(None)
