from args.base_arg_parser import BaseArgParser
from util import str_to_bool


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        self.parser.add_argument('--iters_per_visual', type=int, default=256,
                                 help='Number of iterations between visualizing training examples.')
        self.parser.add_argument('--iters_per_print', type=int, default=4,
                                 help='Number of iterations between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--iters_per_eval', type=int, default=1000,
                                 help='Number of iterations between each model eval/save.')
        self.parser.add_argument('--max_ckpts', type=int, default=5,
                                 help='Maximum number of checkpoints to save.')
        self.parser.add_argument('--metric_name', type=str, default='MSE_src2tgt',
                                 help='Name of metric to determine best checkpoint when saving.')
        self.parser.add_argument('--maximize_metric', type=str_to_bool, default=False,
                                 help='Whether to maximize metric `metric_name` when saving.')
        self.parser.add_argument('--lambda_src', type=float, default=10., help='Source image cycle loss weight.')
        self.parser.add_argument('--lambda_tgt', type=float, default=10., help='Target image cycle loss weight.')
        self.parser.add_argument('--lambda_id', type=float, default=0.5, help='Ratio of loss weights ID:GAN.')
        self.parser.add_argument('--lambda_l1', type=float, default=100., help='Ratio of loss weights L1:GAN.')
        self.parser.add_argument('--lambda_mle', type=float, default=1., help='Ratio of loss weights MLE:GAN.')
        self.parser.add_argument('--beta_1', type=float, default=0.5, help='Adam hyperparameter: beta_1.')
        self.parser.add_argument('--beta_2', type=float, default=0.999, help='Adam hyperparameter: beta_2.')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Adam hyperparameter: initial learning rate.')
        self.parser.add_argument('--rnvp_beta_1', type=float, default=0.5, help='RealNVP Adam hyperparameter: beta_1.')
        self.parser.add_argument('--rnvp_beta_2', type=float, default=0.999, help='RealNVP Adam hyperparameter: beta_2.')
        self.parser.add_argument('--rnvp_lr', type=float, default=2e-4, help='RealNVP learning rate.')
        self.parser.add_argument('--weight_norm_l2', type=float, default=5e-5,
                                 help='L2 regularization factor for weight norm scale factors.')
        self.parser.add_argument('--lr_policy', type=str, default='linear',
                                 help='Learning rate schedule policy. See modules/optim.py for details.',
                                 choices=('linear', 'plateau', 'step'))
        self.parser.add_argument('--lr_step_epochs', type=int, default=100,
                                 help='Number of epochs between each divide-by-10 step (step policy only).')
        self.parser.add_argument('--lr_warmup_epochs', type=int, default=100,
                                 help='Number of epochs before we start decaying the learning rate (linear only).')
        self.parser.add_argument('--lr_decay_epochs', type=int, default=100,
                                 help='Number of epochs to decay the learning rate linearly to 0 (linear only).')
        self.parser.add_argument('--use_mixer', default=True, type=str_to_bool,
                                 help='Use image buffer during training. \
                                      Note that mixer is disabled for conditional GAN by default.')
        self.parser.add_argument('--num_epochs', type=int, default=0,
                                 help='Number of epochs to train. If 0, train forever.')
        self.parser.add_argument('--num_visuals', type=int, default=4, help='Maximum number of visuals per batch.')
        self.parser.add_argument('--clip_gradient', type=float, default=0.,
                                 help='Maximum gradient norm. Setting to 0 disables gradient clipping.')
        self.parser.add_argument('--clamp_jacobian', type=str_to_bool, default=False,
                                 help='Use Jacobian Clamping from https://arxiv.org/abs/1802.08768.')
        self.parser.add_argument('--jc_lambda_min', type=float, default=1.,
                                 help='Jacobian Clamping lambda_min parameter.')
        self.parser.add_argument('--jc_lambda_max', type=float, default=20.,
                                 help='Jacobian Clamping lambda_max parameter.')
