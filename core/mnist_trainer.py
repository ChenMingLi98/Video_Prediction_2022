import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from core.utils import preprocess,metrics
import torch

def train(model,ims,mask_true,configs,itr,file):
    if configs.model_name=='gddn':
        test_discriminator_item_loss = []
        test_pred_item_loss = []
        test_ims=ims.copy()
        f = file
        for n in range(10):
            input_ten_frames=test_ims[:,n:n+configs.total_length]
            img_gen,test_pred_loss,test_discriminator_loss = model.train(input_ten_frames, None)
            test_ims[:,n+configs.total_length-1]=img_gen[:,0]
            test_discriminator_item_loss.append(test_discriminator_loss)
            test_pred_item_loss.append(test_pred_loss)

        if itr % configs.display_interval == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
            f.write('\nitr:%d' % itr)
        return test_pred_item_loss,test_discriminator_item_loss
    else:
        f = file
        cost = model.train(ims, mask_true)
        if configs.reverse_input:
            ims_rev = np.flip(ims, axis=1).copy()
            cost += model.train(ims_rev, mask_true)
            cost = cost / 2

        if itr % configs.display_interval == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
            print('training loss: ' + str(cost))
            f.write('\nitr:%d' % itr)
            f.write('\ntraining loss:%f' % cost)
            #f.close()
        return cost


def test(model,test_input_handle,configs,itr,file,mnist_writer):
    f=file
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path=os.path.join(configs.gen_frm_dir,str(itr//configs.onepoch_interval//4))
    os.mkdir(res_path)
    metrics_results_path=configs.cost_metrics_save_dir+'/metrics/summary.txt'
    file_metrics=open(metrics_results_path,'a')
    avg_mse=0
    batch_id=0
    img_mse,ssim,psnr,fmae,sharp=[],[],[],[],[]
    if configs.model_name=='gddn':
        for i in range(configs.total_length - 1):
            img_mse.append(0)
            ssim.append(0)
            psnr.append(0)
            fmae.append(0)
            sharp.append(0)

        total_difference_test_loss = [0 for i in range(10)]
        total_pred_test_loss = [0 for i in range(10)]
        total_test_loss1 = 0
        total_test_loss2 = 0
        while (test_input_handle.no_batch_left() == False):

            test_difference_item_loss = []
            test_pred_item_loss = []
            next_frames = []
            batch_id += 1
            test_ims = test_input_handle.get_batch()              #[8,20,64,64,1]
            test_dat=test_ims.copy()
            for n in range(10):
                input_ten_frames = test_dat[:, n:n + configs.total_length]
                img_gen, test_difference_loss, test_pred_loss = model.test(input_ten_frames, None)
                test_dat[:, n + configs.total_length - 1] = img_gen[:, 0]
                test_difference_item_loss.append(test_difference_loss)
                test_pred_item_loss.append(test_pred_loss)
                total_difference_test_loss[n] += test_difference_loss
                total_pred_test_loss[n] += test_pred_loss
                total_test_loss1+=test_difference_loss
                total_test_loss2+=test_pred_loss
                next_frames.append(torch.from_numpy(img_gen).squeeze(1))
            next_frames = torch.stack(next_frames, dim=1).contiguous()

            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(batch_id))
            f.write('\nitr:{:02d},batch_id:{:02d}'.format(itr // configs.onepoch_interval//4, batch_id))
            for i in range(configs.input_length):
                print('testing_difference_loss:{:.8f},testing_pred_loss:{:.8f}'.format(test_difference_item_loss[i].item(), test_pred_item_loss[i].item()))
                f.write('\ntesting_difference_loss:{:.8f},testing_pred_loss:{:.8f}'.format(test_difference_item_loss[i].item(), test_pred_item_loss[i].item()))

            img_gen=np.array(next_frames)
            img_gen_length = img_gen.shape[1]
            img_out = img_gen

            # MSE per frame
            for i in range(configs.total_length - 1):
                x = test_ims[:, i + configs.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                fmae[i] += metrics.batch_mae_frame_float(gx, x)
                gx = np.maximum(gx, 0)
                gx = np.minimum(gx, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse

                real_frm = np.uint8(x * 255)
                pred_frm = np.uint8(gx * 255)
                psnr[i] += metrics.batch_psnr(pred_frm, real_frm)
                for b in range(configs.batch_size):
                    sharp[i] += np.max(
                        cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                    score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True, multichannel=True)
                    ssim[i] += score

            # save prediction examples
            if batch_id <= configs.num_save_samples:
                path = os.path.join(res_path, str(batch_id))
                os.mkdir(path)
                for j in range(configs.batch_size):
                    for i in range(configs.total_length + 9):
                        name = str(j + 1) + '_gt' + str(i + 1) + '.jpg'
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(test_ims[j, i, :, :, :] * 255)
                        cv2.imwrite(file_name, img_gt)
                        for i in range(img_gen_length):
                            name = str(j + 1) + '_pd' + str(i + 1 + configs.input_length) + '.jpg'
                            file_name = os.path.join(path, name)
                            img_pd = img_gen[j, i, :, :, :]
                            img_pd = np.maximum(img_pd, 0)
                            img_pd = np.minimum(img_pd, 1)
                            img_pd = np.uint8(img_pd * 255)
                            cv2.imwrite(file_name, img_pd)
            test_input_handle.next()

        for j in range(10):
            print('epoch:{:02d},total_difference_loss:{:.08f},total_pred_loss:{:.08f}'.format(itr // configs.onepoch_interval//4,total_difference_test_loss[j].item(),total_pred_test_loss[j].item()))
            f.write('\nepoch:{:02d},total_difference_loss:{:.08f},total_pred_loss:{:.08f}'.format(itr // configs.onepoch_interval//4,total_difference_test_loss[j].item(),total_pred_test_loss[j].item()))
        mnist_writer.add_scalars('Test Loss', {'pred': total_test_loss2, 'difference': total_test_loss1},global_step=itr//configs.onepoch_interval//4)

        file_metrics.write('\nitr:{:02d}'.format(itr // configs.onepoch_interval//4))
        avg_mse = avg_mse / (batch_id * configs.batch_size)
        print('mse per seq: ' + str(avg_mse))
        file_metrics.write('\nmse per seq: %f' % avg_mse)
        for i in range(configs.total_length-1):
            print(img_mse[i] / (batch_id * configs.batch_size))
            file_metrics.write('\n %f' % (img_mse[i] / (batch_id * configs.batch_size)))

        fmae = np.asarray(fmae, dtype=np.float32) / batch_id
        print('mae per seq:' + str(np.mean(fmae)))
        file_metrics.write('\nfmae per frame: %f' % np.mean(fmae))
        for i in range(configs.total_length -1):
            print(fmae[i])
            file_metrics.write('\n %f' % fmae[i])

        ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
        print('ssim per frame: ' + str(np.mean(ssim)))
        file_metrics.write('\nssim per frame: %f' % np.mean(ssim))
        for i in range(configs.total_length-1):
            print(ssim[i])
            file_metrics.write('\n %f' % ssim[i])

        psnr = np.asarray(psnr, dtype=np.float32) / batch_id
        print('psnr per frame: ' + str(np.mean(psnr)))
        file_metrics.write('\npsnr per frame: %f' % np.mean(psnr))
        for i in range(configs.total_length-1):
            print(psnr[i])
            file_metrics.write('\n %f' % psnr[i])

        sharp = np.asarray(sharp, dtype=np.float32) / (configs.batch_size * batch_id)
        print('sharp per frame:' + str(np.mean(sharp)))
        file_metrics.write('\nsharpness per frame: %f' % np.mean(sharp))
        for i in range(configs.total_length-1):
            print(sharp[i])
            file_metrics.write('\n %f' % sharp[i])

        file_metrics.close()

        return total_test_loss1, total_test_loss2

    else:
        for i in range(configs.total_length - configs.input_length):
            img_mse.append(0)
            ssim.append(0)
            psnr.append(0)
            fmae.append(0)
            sharp.append(0)

        mask_true = np.zeros(
            (configs.batch_size,
             configs.total_length - configs.input_length - 1,
             configs.img_height // configs.patch_size,
             configs.img_width // configs.patch_size,
             configs.patch_size ** 2 * configs.img_channel))

        total_test_loss = 0
        while (test_input_handle.no_batch_left() == False):
            batch_id += 1
            test_ims = test_input_handle.get_batch()
            test_dat = preprocess.reshape_patch(test_ims, configs.patch_size)

            img_gen, test_loss = model.test(test_dat, mask_true)
            total_test_loss += test_loss
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(batch_id))
            print('testing loss: ' + str(test_loss))
            f.write('\nitr:{:02d},batch_id:{:02d}'.format(itr//configs.onepoch_interval//4, batch_id))
            f.write('\ntesting loss:%f' % test_loss)

            img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
            output_length = configs.total_length - configs.input_length
            img_gen_length = img_gen.shape[1]
            img_out = img_gen[:, -output_length:]

            # MSE per frame
            for i in range(configs.total_length - configs.input_length):
                x = test_ims[:, i + configs.input_length, :, :, :]
                gx = img_out[:, i, :, :, :]
                fmae[i] += metrics.batch_mae_frame_float(gx, x)
                gx = np.maximum(gx, 0)
                gx = np.minimum(gx, 1)
                mse = np.square(x - gx).sum()
                img_mse[i] += mse
                avg_mse += mse

                real_frm = np.uint8(x * 255)
                pred_frm = np.uint8(gx * 255)
                psnr[i] += metrics.batch_psnr(pred_frm, real_frm)

                for b in range(configs.batch_size):
                    sharp[i] += np.max(cv2.convertScaleAbs(cv2.Laplacian(pred_frm[b], 3)))
                    score, _ = structural_similarity(pred_frm[b], real_frm[b], full=True, multichannel=True)
                    ssim[i]+=score

            # save prediction examples
                    # save prediction examples
            if batch_id <= configs.num_save_samples:
                path = os.path.join(res_path, str(batch_id))
                os.mkdir(path)
                for j in range(configs.batch_size):
                    for i in range(configs.total_length):
                        name = str(j + 1) + '_gt' + str(i + 1) + '.jpg'
                        file_name = os.path.join(path, name)
                        img_gt = np.uint8(test_ims[j, i, :, :, :] * 255)
                        cv2.imwrite(file_name, img_gt)
                        for i in range(img_gen_length):
                            name = str(j + 1) + '_pd' + str(i + 1 + configs.input_length) + '.jpg'
                            file_name = os.path.join(path, name)
                            img_pd = img_gen[j, i, :, :, :]
                            img_pd = np.maximum(img_pd, 0)
                            img_pd = np.minimum(img_pd, 1)
                            img_pd = np.uint8(img_pd * 255)
                            cv2.imwrite(file_name, img_pd)
            test_input_handle.next()

        print('epoch:{:02d},train_total_train_loss:{:.08f}'.format(itr//configs.onepoch_interval//4, total_test_loss))
        file_metrics.write('\nepoch:{:02d},train_total_train_loss:{:.08f}'.format(itr//configs.onepoch_interval//4, total_test_loss))
        mnist_writer.add_scalar(tag='Test Loss', scalar_value=total_test_loss.item(), global_step=itr//configs.onepoch_interval//4)

        file_metrics.write('\itr:{:02d}'.format(itr//configs.onepoch_interval//4))
        avg_mse = avg_mse / (batch_id * configs.batch_size)
        print('mse per seq: ' + str(avg_mse))
        file_metrics.write('\nmse per seq: %f' % avg_mse)
        for i in range(configs.total_length - configs.input_length):
            print(img_mse[i] / (batch_id * configs.batch_size))
            file_metrics.write('\n %f' % (img_mse[i] / (batch_id * configs.batch_size)))

        fmae = np.asarray(fmae, dtype=np.float32) / batch_id
        print('mae per seq:' + str(np.mean(fmae)))
        file_metrics.write('\nfmae per frame: %f' % np.mean(fmae))
        for i in range(configs.total_length - configs.input_length):
            print(fmae[i])
            file_metrics.write('\n %f' % fmae[i])

        ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * batch_id)
        print('ssim per frame: ' + str(np.mean(ssim)))
        file_metrics.write('\nssim per frame: %f' % np.mean(ssim))
        for i in range(configs.total_length - configs.input_length):
            print(ssim[i])
            file_metrics.write('\n %f' % ssim[i])

        psnr = np.asarray(psnr, dtype=np.float32) / batch_id
        print('psnr per frame: ' + str(np.mean(psnr)))
        file_metrics.write('\npsnr per frame: %f' % np.mean(psnr))
        for i in range(configs.total_length - configs.input_length):
            print(psnr[i])
            file_metrics.write('\n %f' % psnr[i])

        sharp = np.asarray(sharp, dtype=np.float32) / (configs.batch_size * batch_id)
        print('sharp per frame:' + str(np.mean(sharp)))
        file_metrics.write('\nsharpness per frame: %f' % np.mean(sharp))
        for i in range(configs.total_length - configs.input_length):
            print(sharp[i])
            file_metrics.write('\n %f' % sharp[i])

        file_metrics.close()






