import os
import glob

from data.patchdata import PatchData

class Mayo(PatchData):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(rgb_range=1.0)
        parser.add_argument('--thickness', type=int, default=0, choices=[0, 1, 3],
            help='Specify thicknesses of mayo dataset (1 or 3 mm)')
        
        return parser

    def __init__(self, args, name='mayo', is_train=True):
        # Mayo specific
        self.thickness = args.thickness
        super(Mayo, self).__init__(
            args, name=name, is_train=is_train
        )

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[1]))
        )

        return names_hr, names_lr

    def _set_filesystem(self, data_dir):
        super(Mayo, self)._set_filesystem(data_dir)

        if self.thickness == 0:
            full_dose = 'full_*mm'
            quarter_dose = 'quarter_*mm'
            self.dir_hr = os.path.join(self.apath, full_dose)
            self.dir_lr = os.path.join(self.apath, quarter_dose)
            self.ext = ('.tiff', '.tiff')
        else:
            full_dose = 'full_{}mm'.format(self.thickness)
            quarter_dose = 'quarter_{}mm'.format(self.thickness)
            self.dir_hr = os.path.join(self.apath, full_dose)
            self.dir_lr = os.path.join(self.apath, quarter_dose)
            self.ext = ('.tiff', '.tiff')
            