from kaffe.tensorflow import Network


class Pose2d(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')
             .max_pool(2, 2, 2, 2, name='pool1_stage1')
             .conv(3, 3, 128, 1, 1, name='conv2_1')
             .conv(3, 3, 128, 1, 1, name='conv2_2')
             .max_pool(2, 2, 2, 2, name='pool2_stage1')
             .conv(3, 3, 256, 1, 1, name='conv3_1')
             .conv(3, 3, 256, 1, 1, name='conv3_2')
             .conv(3, 3, 256, 1, 1, name='conv3_3')
             .conv(3, 3, 256, 1, 1, name='conv3_4')
             .max_pool(2, 2, 2, 2, name='pool3_stage1')
             .conv(3, 3, 512, 1, 1, name='conv4_1')
             .conv(3, 3, 512, 1, 1, name='conv4_2')
             .conv(3, 3, 256, 1, 1, name='conv4_3_CPM')
             .conv(3, 3, 256, 1, 1, name='conv4_4_CPM')
             .conv(3, 3, 256, 1, 1, name='conv4_5_CPM')
             .conv(3, 3, 256, 1, 1, name='conv4_6_CPM')
             .conv(3, 3, 128, 1, 1, name='conv4_7_CPM'))