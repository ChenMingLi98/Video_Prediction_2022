import os
import torch
import argparse
import numpy as np
import core.kitti_trainer as trainer
from torch.utils.data import DataLoader
from core.data_provider.kitti import KITTI
from core.nets.model_factory import Model
from tensorboardX import SummaryWriter
from core.utils.earlystopping import EarlyStopping


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch driving recorder video frame prediction model - Network Name')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
parser.add_argument('--dataset_name', type=str, default='kitti')
parser.add_argument('--train_data_paths', type=str, default='Input your dataset path')
parser.add_argument('--train_data_sources_paths', type=str, default='Input your dataset path')
parser.add_argument('--valid_data_paths', type=str, default='Input your dataset path')
parser.add_argument('--valid_data_sources_paths', type=str, default='Input your dataset path')
parser.add_argument('--test_data_paths',type=str,default='Input your dataset path')
parser.add_argument('--test_data_sources_paths',type=str,default='Input your dataset path')
parser.add_argument('--save_dir', type=str, default='checkpoints/...')
parser.add_argument('--gen_frm_dir', type=str, default='results/...')
parser.add_argument('--cost_metrics_save_dir',type=str,default='cost_metrics_results/...')
parser.add_argument('--input_length', type=int, default=5)
parser.add_argument('--out_length', type=int, default=5)
parser.add_argument('--total_length', type=int, default=10)
parser.add_argument('--img_height',type=int,default=128)
parser.add_argument('--img_width', type=int, default=160)
parser.add_argument('--img_channel', type=int, default=3)

# model
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='128,128,128,128')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--forget_bias', type=float, default=0.1)
parser.add_argument('--layer_norm',type=int,default=1)


# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_d', type=float, default=0.0002)
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--reverse_input', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=200)
parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--test_interval', type=int, default=2088)
parser.add_argument('--snapshot_interval', type=int, default=2088)
parser.add_argument('--num_save_samples', type=int, default=20)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
print(args)


#创建数据集
train_dataset = KITTI(args.train_data_paths, args.train_data_sources_paths, args.total_length)
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
valid_dataset = KITTI(args.test_data_paths, args.test_data_sources_paths, args.total_length)
validloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

random_seed=1998
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count()>1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


def train_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    #save cost and metrics
    training_cost_path=args.cost_metrics_save_dir+'/train/summary.txt'
    testing_cost_path=args.cost_metrics_save_dir+'/test/summary.txt'
    kitti_train_writer = SummaryWriter(
        log_dir='cost_metrics_results/...')
    kitti_valid_writer=SummaryWriter(
        log_dir='cost_metrics_results/...')

    #手动调整学习率和早停法
    pla_lr_scheduler_difference = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer[0],
                                                                  factor=0.5,
                                                                  patience=5,
                                                                  verbose=True)
    pla_lr_scheduler_pred = torch.optim.lr_scheduler.ReduceLROnPlateau(model.optimizer[1],
                                                                    factor=0.5,
                                                                    patience=5,
                                                                    verbose=True)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    eta = args.sampling_start_value
    total_iteration=0


    for itr in range(1,args.max_iterations+1):
        total_difference_loss=0
        total_pred_loss=0
        file_train = open(training_cost_path, 'a')
        file_train.write('\n----------第{}轮----------'.format(str(itr)))
        for index,data in enumerate(trainloader,1):
            total_iteration+=1
            ims=data                                                      #[8,10,128,160,3]
            loss1,loss2=trainer.train(model,ims,args,total_iteration,file_train)
            total_difference_loss+=loss1
            total_pred_loss+=loss2

            print('epoch:{:02d},train_difference_loss:{:.08f},train_total_pred_loss:{:.08f}'
                  .format(itr,loss1.item(),loss2.item()))
            file_train.write('\nepoch:{:02d},train_difference_loss:{:.08f},train_total_pred_loss:{:.08f}'
                  .format(itr,loss1.item(),loss2.item()))

        print('epoch:{:02d},train_total_difference_loss:{:.08f},avg_difference_loss:{:.08f},train_total_pred_loss:{:.08f},avg_pred_loss:{:.08f}'
                  .format(itr,total_difference_loss,total_difference_loss/len(trainloader),total_pred_loss,total_pred_loss/len(trainloader)))
        file_train.write('\nepoch:{:02d},train_total_difference_loss:{:.08f},avg_difference_loss:{:.08f},train_total_pred_loss:{:.08f},avg_pred_loss:{:.08f}'
                  .format(itr,total_difference_loss,total_difference_loss/len(trainloader),total_pred_loss,total_pred_loss/len(trainloader)))
        file_train.close()
        kitti_train_writer.add_scalar(tag='Train Loss', scalar_value=total_pred_loss.item(),
                                 global_step=itr)

        if total_iteration % args.test_interval == 0:
            file_test = open(testing_cost_path, 'a')
            valid_differnce_loss,valid_pred_loss=trainer.test(model,args, itr,validloader,file_test,kitti_valid_writer)
            pla_lr_scheduler_difference.step(valid_differnce_loss)
            pla_lr_scheduler_pred.step(valid_pred_loss)
            file_test.close()

        if total_iteration % args.snapshot_interval == 0:
            if total_iteration<=50000:
                model.save(itr)
            else:
                early_stopping(valid_pred_loss.item(), model,itr)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    break

def test_wrapper(model):
    model.load(args.pretrained_model)
    test_file_list = os.listdir(args.test_data_paths)
    testing_cost_path = args.cost_metrics_save_dir + '/test/final.txt'
    file_test = open(testing_cost_path, 'a')
    d2city_test_writer = SummaryWriter(
        log_dir='cost_metrics_results/...')
    test_loss = trainer.test(model, args,4, len(test_file_list), file_test, d2city_test_writer)
    file_test.close()


print('Initializing models')

model=Model(args)
if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)







