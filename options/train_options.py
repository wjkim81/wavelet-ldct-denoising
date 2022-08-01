from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # network saving and loading parameters
        # parser.add_argument('--save_latest_freq', type=int, default=5000,
        #     help='frequency of saving the latest results')
        parser.add_argument('--epoch', type=int, default=1,
            help='current epoch')
        parser.add_argument('--n_epochs', type=int, default=200,
            help='number of epochs of training')
        parser.add_argument('--best_psnr', type=float, default=0,
            help='best valid psnr')

        parser.add_argument('--save_epoch_freq', type=int, default=5,
            help='frequency of saving checkpoints at the end of epochs, -1: no save')
        parser.add_argument('--save_test_freq', type=int, default=5,
            help='frequency of saving test result image at the end of epochs, -1: no save')
        parser.add_argument('--n_saves', type=int, default=1,
            help='number of images to save')

        parser.add_argument('--exprdir', type=str, default='experiment_name',
            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--save_spectrum', action='store_true',
            help='save subband spectrum')
        
        parser.add_argument('--test_random_patch', dest='test_random_patch', action='store_true',
            help='test random extracted patch from image')
        parser.add_argument('--no_test_random_patch', dest='test_random_patch', action='store_false',
            help='test random extracted patch from image')
        parser.set_defaults(test_random_patch=True)

        parser.add_argument('--valid_ratio', type=float, default=0.05,
            help='reduce test dataset with ratio')
        parser.add_argument('--test_ratio', type=float, default=1.0,
            help='reduce test dataset with ratio')

        # parser.add_argument('--phase', type=str, default='train',
        #     help='train, valid, test, etc')

        parser.add_argument('--test_every', type=int, default=1000,
            help='do test per every N batches')
        parser.add_argument('--augment', dest='augment', action='store_true',
            help='do random flip (vertical, horizontal, rotation)')
        parser.add_argument('--no_augment', dest='augment', action='store_false',
            help='do not random flip')
        parser.set_defaults(augment=True)
        
        # training parameters
        parser.add_argument('--init_lr_n_epochs', type=int, default=100,
            help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100,
            help='number of epochs to linearly decay learning rate to zero')

        parser.add_argument('--lr', type=float, default=0.0002,
            help='initial learning rate for adam')
        parser.add_argument('--b1', type=float, default=0.9,
            help='Adam: decay of first order momentum of gradient')
        parser.add_argument('--b2', type=float, default=0.999,
            help='Adam: decay of second order momentum of gradient')
        parser.add_argument('--lr_policy', type=str, default='plateau',
            choices=['linear', 'step', 'plateau', 'cosine'],
            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--n_patiences', type=int, default=10,
            help='number of time of patiens of plateau policy')

        parser.add_argument('--log_file', type=str, default='log.csv',
            help='log file to keep track losses of each epoch')

        parser.set_defaults(test_patches=True)

        self.is_train = True
        return parser