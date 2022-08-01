import os
import time
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.tester import test_net_by_tensor_patches, calc_metrics, calc_ssim, \
                         save_tensors, save_metrics, save_summary

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    opt.is_train = False
    opt.test_random_patch = False
    opt.test_ratio = 1.0
    opt.multi_gpu = 0
    # hard-code some parameters for test
    opt.n_threads = 0   # test code only supports num_threads = 1

    # opt.test_patches = True if 'wavelet' in opt.model else False

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model.eval()
    start_time = time.time()

    for di, test_dataloader in enumerate(dataloader['test']):
        print("*** Test phase ***")
        if len(test_dataloader) == 0: continue
        avg_loss = 0.0
        avg_psnr = 0.0
        noise_avg_loss = 0.0
        noise_avg_psnr = 0.0
        avg_ssim = 0.0
        noise_avg_ssim = 0.0

        for i, batch in enumerate(test_dataloader, 1):
            x, target, filename = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if opt.test_patches:
                    out = test_net_by_tensor_patches(opt, model, x)
                    # Make sure every tensor variable is in the same device and detached
                else:
                    input = {
                        'x': x,
                        'target': target
                    }
                    model.set_input(input)
                    model.test()
                    out = model.out
            
            x = x.to(opt.device).detach()
            out = out.to(opt.device).detach()
            target = target.to(opt.device).detach()
            if 'yonsei' in opt.test_datasets[0]:
                x = x * 5
                out = out * 5
                target = target * 5

            # Show and save the results when it is testing phase
            tensors_dict = {
                "x": x,
                "out": out,
                "target": target,
                "filename": filename
            }
            noise_loss, noise_psnr, batch_loss, batch_psnr = calc_metrics(tensors_dict)
            noise_ssim, batch_ssim = calc_ssim(tensors_dict)

            noise_avg_loss += noise_loss
            noise_avg_psnr += noise_psnr
            avg_loss += batch_loss
            avg_psnr += batch_psnr

            avg_ssim += batch_ssim
            noise_avg_ssim += noise_ssim

            end_time = time.time()
            print("** Test {:.3f}s => Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}".format(
                end_time - start_time, i, len(test_dataloader), noise_loss.item(), noise_psnr.item(), batch_loss.item(), batch_psnr.item()
            ))
            print("** Test Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
                noise_avg_loss / i, noise_avg_psnr / i, avg_loss / i, avg_psnr / i
            ))
            print("** SSIM => Noise SSIM: {:.8f}, SSIM: {:.8f}, Average Noise SSIM: {:.8f}, Average SSIM: {:.8f}".format(
                noise_ssim, batch_ssim, noise_avg_ssim / i, avg_ssim / i
            ))
            save_tensors(opt, di, tensors_dict)
            save_metrics(opt, di, i, filename, noise_loss, noise_psnr, batch_loss, batch_psnr)

        avg_loss, avg_psnr = avg_loss / i, avg_psnr / i
        noise_avg_loss, noise_avg_psnr = noise_avg_loss / i, noise_avg_psnr / i
        noise_ssim, avg_ssim = noise_avg_ssim / i, avg_ssim / i

        print("===> Test on {} - Noise Average Loss: {:.8f}, Noise Average PSNR: {:.8f}, Noise Average SSIM: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}, Average SSIM: {:.8f}".format(
            opt.test_datasets[di], noise_avg_loss, noise_avg_psnr, noise_ssim, avg_loss, avg_psnr, avg_ssim
        ))
        save_summary(opt, di, noise_avg_loss, noise_avg_psnr, noise_ssim, avg_loss, avg_psnr, avg_ssim)