import argparse


# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# ----------------------------------------
# General Arguments
general_arg = add_argument_group("General")

general_arg.add_argument("--data_dir", type=str,
                       default="./data",
                       help="Directory with of Human3.6M HDF5 Data")

general_arg.add_argument('--manual-seed', type=int, default=-1,
                         help='Manually set seed')

general_arg.add_argument('--gpu', type=str2bool, default=False,
                         help='Prefer GPU Usage')

# ----------------------------------------
# Model Arguments
model_arg = add_argument_group("Model")

model_arg.add_argument('--model_path', type=str, default='',
                       help='Provide a full path to a previously trained model')
model_arg.add_argument('--continue', type=str2bool, default=False,
                       help='Continue training from last point')

# ----------------------------------------
# Hyperparameter Arguments
hyperparameter_arg = add_argument_group('Hyperparameters')
hyperparameter_arg.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
hyperparameter_arg.add_argument('--lr-decay', type=float, default=0.0, help='Learning Rate Decay')
hyperparameter_arg.add_argument('--momentum', type=float, default=0.0, help='Momentum')
hyperparameter_arg.add_argument('--weight-decay', type=float, default=0.0, help='Weight Decay')
hyperparameter_arg.add_argument('--optim', type=str, choices=('adam', 'rmsprop', 'sgd', 'nag', 'adadelta'),
                                default='adam', help='Optimization Method')
# ----------------------------------------
# Training Arguments
training_arg = add_argument_group('Training')
training_arg.add_argument('--n_epochs', type=int, default=100, help='Total number of epochs to run')
training_arg.add_argument('--nt_iters', type=int, help='Number of training iterations per epoch')
training_arg.add_argument('--t_batch_size', type=int, default=3, help='Minibatch size')
training_arg.add_argument('--nv_iters', type=int, help='Number of validation iterations per epoch')
training_arg.add_argument('--v_batch_size', type=int, default=1, help='Validation minibatch size')

# ----------------------------------------
# Data Arguments
data_arg = add_argument_group('Data')
data_arg.add_argument('--input_res', default=368, type=int, help='Input image resolution')
data_arg.add_argument('--label_size', default=51, type=int, help='Output heatmap resolution')
data_arg.add_argument('--train_h5_path', type=str, help='Name of training data file')
data_arg.add_argument('--valid_h5_path', type=str, help='Name of validation data file')
data_arg.add_argument('--min_scale', type=float, default=0.9)
data_arg.add_argument('--max_scale', type=float, default=1.1)
data_arg.add_argument('--no_subtract_mean', default=False, type=str2bool, help='Whether to subtract 0.5 to [-0.5, 0.5]')

# ----------------------------------------
# RPSM Arguments
rpsm_arg = add_argument_group('RPSM')
rpsm_arg.add_argument('--rho', default=3, type=int, help='Number of cpm joints')
rpsm_arg.add_argument('--hidden_size', default=2048, type=int, help='Number of joints')
rpsm_arg.add_argument('--feat_ind', default=33, type=int, help='Index of the shared 2d pose model')
rpsm_arg.add_argument('--num_ch', default=128, type=int, help='Image feature channel number')
rpsm_arg.add_argument('--np', default=17, type=int, help='Number of joints')
rpsm_arg.add_argument('--img_feat_dim', default=2048, type=int,
                      help='The coarse feature extract image feature dimension')
rpsm_arg.add_argument('--shared_model', type=str, help='Shared model in 2D pose module')

# ----------------------------------------
# Temporal Pose Estimation Arguments
tpe_arg = add_argument_group('Temporal Pose Estimation')
tpe_arg.add_argument('--max_frames', default=20, type=int, help='Max frames to  capture information')
tpe_arg.add_argument('--root_image_folder', type=str)
tpe_arg.add_argument('--train_image_list', type=str)
tpe_arg.add_argument('--valid_image-list', type=str)

def get_config():
    config, unparsed = parser.parse_known_args()

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
