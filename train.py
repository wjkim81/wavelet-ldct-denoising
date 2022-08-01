import time
import torch


from utils.trainer import train_net, valid_net
from data import create_dataset
from models import create_model

from options.train_options import TrainOptions
# from utils.trainer import save_random_tensors


if __name__ == '__main__':
    opt = TrainOptions().parse()
    torch.manual_seed(opt.seed)

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataloader['train'])    # get the number of images in the dataset.
    print('\nThe number of training iteration = {}\n'.format(dataset_size))

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    best_psnr = opt.best_psnr
    start_epoch = opt.epoch
    if start_epoch == 1:
        with open(opt.log_file, mode='w') as f:
            f.write("epoch,train_loss,train_psnr,valid_loss,valid_psnr\n")

    for epoch in range(start_epoch, opt.n_epochs + 1):
        train_loss, train_psnr = train_net(opt, model, dataloader['train'])
        valid_loss, valid_psnr = train_net(opt, model, dataloader['test'], train=False)

        print('saving the latest model (epoch {}, total_iters {})'.format(epoch, opt.n_epochs))
        model.remove_networks('latest')
        model.save_networks('latest', epoch, valid_loss, valid_psnr)
        
        if opt.save_epoch_freq != -1 and epoch % opt.save_epoch_freq == 0:
            print('saving the model (epoch {}, total_iters {}) with frequency {}'.format(epoch, opt.n_epochs, opt.save_epoch_freq))
            model.save_networks(epoch, epoch, valid_loss, valid_psnr)

        # if opt.save_freq != -1 and epoch % opt.save_freq == 0:
        #     print('saving random test results in {}'.format(opt.exprdir))
        #     save_random_tensors(opt, epoch, model, dataloader['test'])

        if valid_psnr > best_psnr:
            print('saving the best model (epoch {}, total_iters {})'.format(epoch, opt.n_epochs))
            model.remove_networks('best')
            model.save_networks('best', epoch, valid_loss, valid_psnr)
            best_psnr = valid_psnr

        with open(opt.log_file, mode='a') as f:
            f.write("{},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
                epoch, train_loss, train_psnr, valid_loss, valid_psnr
            ))

        model.update_learning_rate(valid_loss)
        opt.epoch = epoch + 1