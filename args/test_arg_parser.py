from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--num_examples', type=int, default=-1, help='Number of examples.')
        self.parser.add_argument('--results_dir', type=str, default='results/', help='Save dir for test results.')
        self.parser.add_argument('--save_images', type=str, default=None, choices=(None, 'tensorboard', 'disk'),
                                 help='Where to save images.')
