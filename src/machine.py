import os

import tensorflow as tf

from util.config import get_config, print_usage
"Original base for the code from CSC486B/CSC586B by Kwang Moo Yi, released under the MIT license"

class Machine(object):
    """Network class """

    def __init__(self, x_shp, config):

        self.config = config

        # Get shape
        self.x_shp = x_shp

        # Build the network
        self._build_placeholder()
        self._build_2d_pose_module()
        self._build_feature_adaption_module()
        self._build_3d_pose_recurrent_module()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _build_placeholder(self):
        """Build placeholders."""
        self.x_in = tf.placeholder()
        self.y_in = tf.placeholder()
        self.in_2d_pose = tf.placeholder() # 2d pose module 46x46x128 output from prev stage
        self.in_3d_pose_fc = tf.placeholder() # output from 3D Pose Recurrent Module from prev stage
        self.t = tf.placeholder() # the step

    def _build_2d_pose_module(self):
        with tf.variable_scope("2dPoseModule", reuse=tf.AUTO_REUSE):
            # insert concat op between conv4_5 and out_2d_pose_in
            # if t == 0 skip concat?
            # then do conv4 and conv4 7
            self.out_2d_pose_out = None # TODO output from this module

    def _build_feature_adaption_module(self):
        with tf.variable_scope("FeatureAdaptionModule", reuse=tf.AUTO_REUSE):
            input = self.out_2d_pose_out
            self.feature_adaption_output = None #TODO 1024x1


    def _build_3d_pose_recurrent_module(self):
        with tf.variable_scope("3dPoseRecurrentModule", reuse=tf.AUTO_REUSE):
            self.input = self.feature_adaption_output
            # TODO concat with in_3d_pose_fc
            # if t=0 skip concat?
            self.out_3d_pose = None #TODO

    def _build_loss(self):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            self.loss = None # TODO which is defined as the Euclidean distances between the prediction for all P joints and ground truth

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
                learning_rate=self.config.learning_rate)
            self.optim = optimizer.minimize(
                self.loss, global_step=self.global_step)

    def _build_eval(self):
        """Build the evaluation related ops"""

        with tf.variable_scope("Eval", tf.AUTO_REUSE):
            # Record summary for accuracy
            tf.summary.scalar("accuracy", self.acc)

            self.best_va_acc_in = tf.placeholder(tf.float32, shape=())
            self.best_va_acc = tf.get_variable("best_va_acc", shape=(), trainable=False)
            self.acc_assign_op = tf.assign(self.best_va_acc, self.best_va_acc_in)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create summary writers (one for train, one for validation)
        self.summary_tr = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "train"))
        self.summary_va = tf.summary.FileWriter(
            os.path.join(self.config.log_dir, "valid"))
        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")

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


        # Run TensorFlow Session
        with tf.Session() as sess:
            # Init
            print("Initializing...")
            sess.run(tf.global_variables_initializer())

            # TODO: Check if previous train exists
            b_resume = tf.train.latest_checkpoint(self.save_file_cur)
            if b_resume:
                # TODO: Restore network
                print("Restoring from {}...".format(
                    self.config.log_dir))
                self.saver_cur.restore(sess, self.save_file_cur)
                # TODO: restore number of steps so far
                step = tf.get_variable("step", shape=1)
                # TODO: restore best acc
                best_acc = tf.get_variable("best_va_acc", shape=1)
            else:
                print("Starting from scratch...")
                step = 0
                best_acc = 0

            print("Training...")
            # TODO fetch self.out_2d_pose and self.out_3d_pose_fc and use for self.in_2d_pose and self.in_3d_pose for
            # placholders for the next step

    def test(self, x_te, y_te):
        """Test routine"""

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
