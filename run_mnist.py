import argparse
import numpy as np
from core.data_provider import datasets_factory
from core.nets.model_factory import Model
from core.utils import preprocess
import core.mnist_trainer as trainer
from tensorboardX import SummaryWriter

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch driving recorder video frame prediction model - Network Name')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='Input your dataset path')
parser.add_argument('--valid_data_paths', type=str, default='Input your dataset path')
parser.add_argument('--save_dir', type=str, default='checkpoints/...')
parser.add_argument('--gen_frm_dir', type=str, default='results/...')
parser.add_argument('--cost_metrics_save_dir',type=str,default='cost_metrics_results/...')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_height',type=int,default=64)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--forget_bias', type=float, default=0.1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm',type=int,default=1)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=80000)
parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--onepoch_interval', type=int, default=1250)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

args = parser.parse_args()
print(args)

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_height // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_height // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    mask_true = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                mask_true.append(ones)
            else:
                mask_true.append(zeros)
    mask_true = np.array(mask_true)
    mask_true = np.reshape(mask_true,(args.batch_size,
                                      args.total_length - args.input_length - 1,
                                      args.img_height // args.patch_size,
                                      args.img_width // args.patch_size,
                                      args.patch_size ** 2 * args.img_channel))
    return eta, mask_true

#创建数据集

def train_wrapper(model):
    #load data
    train_input_handle,test_input_handle=datasets_factory.data_provider(
        args.dataset_name,args.train_data_paths,args.valid_data_paths,args.batch_size,args.img_width,
        seq_length=args.total_length,is_training=True)

    eta=args.sampling_start_value

    mnist_train_writer = SummaryWriter(
        log_dir='cost_metrics_results/...')
    mnist_valid_writer = SummaryWriter(
        log_dir='cost_metrics_results/...')

    #save cost and metrics
    training_cost_path=args.cost_metrics_save_dir+'/train/summary.txt'
    testing_cost_path=args.cost_metrics_save_dir+'/test/summary.txt'


    total_train_loss=0
    for itr in range(1,args.max_iterations+1):


        file_train = open(training_cost_path, 'a')
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)
        ims=train_input_handle.get_batch()
        ims=preprocess.reshape_patch(ims,args.patch_size)

        eta,mask_true=schedule_sampling(eta,itr)

        loss=trainer.train(model,ims,mask_true,args,itr,file_train)
        total_train_loss+=loss

        if itr%args.onepoch_interval==0:
            print('epoch:{:02d},train_total_train_loss:{:.08f}'.format(itr//args.onepoch_interval, total_train_loss))
            file_train.write('\nepoch:{:02d},train_total_train_loss:{:.08f}'.format(itr//args.onepoch_interval, total_train_loss))
            mnist_train_writer.add_scalar(tag='Train Loss', scalar_value=total_train_loss.item(),global_step=itr)
            total_train_loss = 0


        if itr%args.snapshot_interval==0:
            model.save(itr)

        if itr%args.test_interval==0:
            file_test=open(testing_cost_path,'a')
            trainer.test(model,test_input_handle,args,itr,file_test,mnist_valid_writer)
            file_test.close()

        file_train.close()
        train_input_handle.next()


print('Initializing models')

model=Model(args)
train_wrapper(model)







