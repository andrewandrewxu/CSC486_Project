import matplotlib.pyplot as plt
import tensorflow as tf
from pretrained.Pose2dFeatureExtraction import Pose2d
import numpy as np
import tqdm
import os

"Original base for the code from CSC486B/CSC586B by Kwang Moo Yi, released under the MIT license"


class RPSMNetwork(object):
    """Network class """

    def __init__(self, train_dataset, val_dataset, config):

        self.config = config
        self.num_states = 1024
        self.image_size = 368
        self.num_output = 51
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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
            hidden_state = tf.get_variable('hidden_state', dtype=tf.float32, shape=(1, 1024),
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

            self.best_va_loss_in = tf.placeholder(tf.float32, shape=())
            self.best_va_loss = tf.get_variable(
                "best_loss_acc", shape=(), trainable=False)
            # Assign op to store this value to TF variable
            self.loss_assign_op = tf.assign(
                self.best_va_loss, self.best_va_loss_in)

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
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()

        self.save_file_cur = os.path.join(
            self.config.log_dir, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.config.save_dir, "model")

    def resume_if_checkpoint(self, sess):
        b_resume = tf.train.latest_checkpoint(self.save_file_cur)
        if b_resume:
            print("Restoring from {}...".format(
                self.config.log_dir))
            self.saver_cur.restore(sess, self.save_file_cur)
            self.step = sess.run(self.global_step)
            self.best_loss = sess.run(self.best_va_loss)
        else:
            print("Starting from scratch...")
            self.step = 0
            self.best_loss = 0

    def validate(self, res, sess):
        if res["loss"] > self.best_loss:
            #Not sure if we have to do tf.assign here, I commented mine below
            #tf.assign(self.best_loss, res["loss"])
            self.best_loss = res["loss"]
            
            sess.run(

                fetches={
                    "loss": self.loss_assign_op,
                    "summary": self.summary_op,
                    "global_step": self.global_step,
                },
                
                feed_dict={
                    self.best_va_loss_in: self.best_loss
                })

            #Write Summary here
            self.summary_va.add_summary(
                res["summary"], global_step=res["global_step"],
            )
            # Save the best model
            self.saver_best.save(
                sess, self.save_file_best,
                write_meta_graph=False,
            )

    def show_result_2d(self, img, pred, truth):
        plt.imshow(img)  # why is it only showing the blue channel??
        pred = np.reshape(pred, (17, 3))
        exp = np.reshape(truth, (17, 3))
        pred *= self.image_size
        exp *= self.image_size
        plt.scatter(x=pred[:, 0], y=pred[:, 1], c='r')
        plt.scatter(x=truth[:, 0], y=truth[:, 1], c='b')
        plt.show()

    def train(self, opts):
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
            pose2d.load('pretrained{}Pose2dFeatureExtraction.npy'.format(os.sep), sess)

            self.resume_if_checkpoint(sess)

            print("Training...")
            for iter in tqdm.trange(opts.nt_iters):
                self._init_recurrent()

                x_tr, y_tr = self.train_dataset.get(index=iter)

                features = sess.run(pose2d.get_output(), feed_dict={self.x_in: x_tr})
                i=0
                for feature, label in zip(features, y_tr):
                    res = sess.run(
                        fetches={
                            "optim": self.optim,
                            "pred": self.pred,
                            "loss": self.loss,
                            "summary": self.summary_op,
                            "global_step": self.global_step,
                            "2dout": self.output_2d_assign,  # assign 2d pose features used in next step
                            "3dout": self.out_pred_assign,  # assign 3d pose used in next step
                            "state": self.state_assign  # assign state used for next step
                        },
                        feed_dict={
                            self.features_in: feature,
                            self.y_in: label
                        },
                    )
                    if i == len(x_tr) - 1:
                        pass
                        # self.show_result_2d(x_tr[i], res["pred"], label)
                    i+=1

                # Write Training Summary
                self.summary_tr.add_summary(
                    res["summary"], global_step=res["global_step"],
                )

                self.saver_cur.save(
                    sess, self.save_file_cur,
                    global_step=self.global_step,
                    write_meta_graph=False,
                )

                #Validation 

                # hardcoded 500 iterations, not sure if need validate every 5000 iterations
                b_validate = iter % 500 == 0
                if b_validate:

                    self.validate(res,sess)

                    # self.summary_va.flush()

                # self.validate(res, sess)

            #not sure if we need to save_cur again here, I just leave it
            self.saver_cur.save(sess, self.save_file_cur)
            self.summary_tr.add_graph(sess.graph)
            self.summary_va.add_graph(sess.graph)


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
                    "loss": self.loss
               },
               feed_dict={
                    self.x_in: x_te,
                    self.y_in: y_te,
               },
            )
            print("Test accuracy is {}".format(res["loss"]))
