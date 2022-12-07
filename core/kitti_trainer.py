import os.path
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from core.utils import metrics


def train(model,ims,mask_true,configs,itr,file):
    if configs.model_name=='gddn':
        f = file
        cost1,cost2= model.train(ims, mask_true)
        if itr % configs.display_interval == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
            f.write('\nitr:%d' % itr)
        return cost1,cost2



def test(model,configs,itr,testloader,file,kitti_writer):
    f=file
    res_path = os.path.join(configs.gen_frm_dir, str(itr//4))
    os.mkdir(res_path)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'test...')
    metrics_results_path=configs.cost_metrics_save_dir+'/metrics/summary.txt'
    file_metrics=open(metrics_results_path,'a')
    avg_mse=0
    img_mse,ssim,psnr,fmae,sharp=[],[],[],[],[]
    for i in range(configs.total_length-configs.input_length):
        img_mse.append(0)
        ssim.append(0)
        psnr.append(0)
        fmae.append(0)
        sharp.append(0)

    mask_true=np.zeros(
        (configs.batch_size,
         configs.total_length-configs.input_length-1,
         configs.img_height//configs.patch_size,
         configs.img_width//configs.patch_size,
         128))
    total_num=0
    total_difference_loss=0
    total_pred_loss=0
    batch_id=0

    for index, data in enumerate(testloader, 1):
        total_num+=1
        batch_id+=1
        test_ims=data          #[4,10,128,160,3]
        test_dat=test_ims

        img_gen, test_difference_loss,test_pred_loss = model.test(test_dat, mask_true)
        total_difference_loss+=test_difference_loss
        total_pred_loss+=test_pred_loss
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(total_num))
        print('testing_difference_loss:{:.8f},testing_pred_loss:{:.8f}'.format(test_difference_loss,test_pred_loss))
        f.write('\nitr:{:02d},batch_id:{:02d}'.format(itr//4,total_num))
        f.write('\ntesting_difference_loss:{:.8f},testing_pred_loss:{:.8f}'.format(test_difference_loss,test_pred_loss))

        img_gen_length = img_gen.shape[1]
        img_out=img_gen

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

            real_frm = np.uint8((x+1) * 127.5)
            pred_frm = np.uint8((gx+1) * 127.5)
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
            for i in range(configs.total_length):
                name = 'gt' + str(i + 1) + '.png'
                file_name = os.path.join(path, name)
                img_gt = np.uint8((test_ims[0, i, :, :, :]+1)*127.5)
                cv2.imwrite(file_name, img_gt)
                for i in range(img_gen_length):
                    name = 'pd' + str(i + 1 + configs.input_length) + '.png'
                    file_name = os.path.join(path, name)
                    img_pd = img_gen[0, i, :, :, :]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8((img_pd+1)*127.5)
                    cv2.imwrite(file_name, img_pd)

    print('epoch:{:02d},total_difference_loss:{:.08f},total_pred_loss:{:.08f}'.format(itr//4,total_difference_loss.item(),total_pred_loss.item()))
    f.write('\nepoch:{:02d},total_difference_loss:{:.08f},total_pred_loss:{:.08f}'.format(itr//4,total_difference_loss.item(),total_pred_loss.item()))
    kitti_writer.add_scalar(
        tag='Test Loss', scalar_value=total_pred_loss.item(),global_step=itr//4
    )





    file_metrics.write('\nitr:{:02d}'.format(itr//4))
    avg_mse = avg_mse / (total_num * configs.batch_size)
    print('mse per seq: ' + str(avg_mse))
    file_metrics.write('\nmse per seq: %f' % avg_mse)
    for i in range(configs.total_length - configs.input_length):
        print(img_mse[i] / (total_num * configs.batch_size))
        file_metrics.write('\n %f' % (img_mse[i] / (total_num * configs.batch_size)))

    fmae=np.asarray(fmae,dtype=np.float32)/total_num
    print('mae per seq:'+str(np.mean(fmae)))
    file_metrics.write('\nfmae per frame: %f' % np.mean(fmae))
    for i in range(configs.total_length-configs.input_length):
        print(fmae[i])
        file_metrics.write('\n %f' % fmae[i])

    ssim = np.asarray(ssim, dtype=np.float32) / (configs.batch_size * total_num)
    print('ssim per frame: ' + str(np.mean(ssim)))
    file_metrics.write('\nssim per frame: %f' % np.mean(ssim))
    for i in range(configs.total_length - configs.input_length):
        print(ssim[i])
        file_metrics.write('\n %f' % ssim[i])

    psnr = np.asarray(psnr, dtype=np.float32) / total_num
    print('psnr per frame: ' + str(np.mean(psnr)))
    file_metrics.write('\npsnr per frame: %f' % np.mean(psnr))
    for i in range(configs.total_length - configs.input_length):
        print(psnr[i])
        file_metrics.write('\n %f' % psnr[i])

    sharp=np.asarray(sharp,dtype=np.float32)/(configs.batch_size*total_num)
    print('sharp per frame:'+str(np.mean(sharp)))
    file_metrics.write('\nsharpness per frame: %f' % np.mean(sharp))
    for i in range(configs.total_length-configs.input_length):
        print(sharp[i])
        file_metrics.write('\n %f' % sharp[i])

    file_metrics.close()

    return total_difference_loss,total_pred_loss




