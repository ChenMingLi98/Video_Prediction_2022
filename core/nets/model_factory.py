import torch
import torch.nn as nn
from torch.optim import Adam
from core.nets import ConvLstm,PredRnn,PredRnn_pp,MIM,MotionRNN,Eidetic3DLSTM,SA_ConvLSTM,GDDN,C3D_Discriminator


class Model(object):
    def __init__(self,configs):
        self.configs=configs
        self.hidden_dim=[int(x) for x in configs.num_hidden.split(',')]
        self.num_layers=len(self.hidden_dim)
        self.adversarial_loss=nn.MSELoss().to(configs.device)
        networks_map={
            'convlstm':ConvLstm.ConvLstm,
            'predrnn':PredRnn.RNN,
            'predrnn_pp':PredRnn_pp.RNN,
            'mim':MIM.RNN,
            'motionrnn':MotionRNN.MotionRNN,
            'e3d':Eidetic3DLSTM.Eidetic3DLSTM,
            'saconvlstm':SA_ConvLSTM.SAConvLstm,
            'gddn':GDDN
        }

        if configs.model_name in networks_map and configs.model_name!='gddn':
            Network=networks_map[configs.model_name]
            self.network=Network(self.hidden_dim,self.num_layers,configs).to(configs.device)
            self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        elif configs.model_name=='gddn':
            self.pred_network=GDDN.Predictor(configs).to(configs.device)
            self.pred_difference_frame_network=GDDN.Motion_Context(configs).to(configs.device)
            self.discriminator=C3D_Discriminator.C3D(configs).to(configs.device)
            self.optimizer =(Adam([{'params':self.pred_difference_frame_network.parameters(),'lr':configs.lr},
                                   {'params':self.pred_network.parameters(),'lr':configs.lr}]),
                             Adam(self.discriminator.parameters(),lr=configs.lr_d, betas=(0.9,0.999)))
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)



    def save(self, itr):
        if self.configs.model_name=='gddn':
            checkpoint_difference_path = 'Input your model_train_param path'
            checkpoint_pred_path = 'Input your model_train_param path'
            checkpoint_discriminator_path = 'Input your model_train_param path'
            torch.save(self.pred_difference_frame_network.state_dict(), checkpoint_difference_path)
            torch.save(self.pred_network.state_dict(), checkpoint_pred_path)
            torch.save(self.discriminator.state_dict(), checkpoint_discriminator_path)
            print("save difference model to %s" % checkpoint_difference_path)
            print("save pred model to %s" % checkpoint_pred_path)
            print("save discriminator model to %s" % checkpoint_discriminator_path)
        else:
            stats = {}
            stats['net_param'] = self.network.state_dict()
            checkpoint_path = 'Input your model_train_param path'
            torch.save(stats, checkpoint_path)
            print("save model to %s" % checkpoint_path)


    def load(self, checkpoint_pred_path,checkpoint_difference_path,checkpoint_discriminator_path):
        if checkpoint_discriminator_path!=None:
            print('load model:', checkpoint_pred_path, checkpoint_difference_path, checkpoint_discriminator_path)

            stats_pred = torch.load(checkpoint_pred_path)
            stats_difference = torch.load(checkpoint_difference_path)
            stats_discriminator = torch.load(checkpoint_discriminator_path)
            self.pred_network.load_state_dict(stats_pred)
            self.pred_difference_frame_network.load_state_dict(stats_difference)
            self.discriminator.load_state_dict(stats_discriminator)
        else:
            print('load model:', checkpoint_pred_path, checkpoint_difference_path)
            stats_pred = torch.load(checkpoint_pred_path)
            stats_difference = torch.load(checkpoint_difference_path)
            self.pred_network.load_state_dict(stats_pred)
            self.pred_difference_frame_network.load_state_dict(stats_difference)



    def train(self,input,mask_true):
        if self.configs.model_name=='gddn':
            input_tensor = torch.FloatTensor(input).to(self.configs.device)              #[8,5,128,160,3]

            difference_x = input_tensor[:, 1:, :, :, :] - input_tensor[:, :-1, :, :, :]  # [8,4,128,160,3]
            self.optimizer[0].zero_grad()
            next_differences,loss_difference = self.pred_difference_frame_network(difference_x,self.configs.out_length)

            next_frames, loss_pred=self.pred_network(input_tensor, self.configs.out_length,difference_x,
                              next_differences[0], next_differences[1])
            input_seq=torch.cat((input_tensor[:,:self.configs.input_length],next_frames),1)
            D_result_fake = self.discriminator(input_seq.permute(0,4,1,2,3).contiguous()).reshape(-1)
            loss_p1 = loss_pred + self.configs.lambda_difference*loss_difference+self.configs.lambda_seq_GAN*self.adversarial_loss(D_result_fake,torch.ones_like(D_result_fake))
            loss_p1.backward()
            self.optimizer[0].step()

            # train Discriminator
            self.optimizer[1].zero_grad()
            d_real = self.discriminator(input_tensor.permute(0,4,1,2,3).contiguous()).reshape(-1)
            real_loss = self.adversarial_loss(d_real, torch.ones_like(d_real))
            d_fake = self.discriminator(input_seq.detach().permute(0,4,1,2,3).contiguous()).reshape(-1)
            fake_loss = self.adversarial_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            self.optimizer[1].step()
            #return next_frames.detach().cpu().numpy(),loss_p1.detach().cpu().numpy(), d_loss.detach().cpu().numpy()
            return loss_p1.detach().cpu().numpy(), d_loss.detach().cpu().numpy()

        else:
            input_tensor = torch.FloatTensor(input).to(self.configs.device)
            mask_true_tensor = torch.FloatTensor(mask_true).to(self.configs.device)
            self.optimizer.zero_grad()
            next_frames, loss = self.network(input_tensor, mask_true_tensor)
            loss.backward()
            self.optimizer.step()
            return loss.detach().cpu().numpy()


    def test(self,input,mask_true):
        if self.configs.model_name=='gddn':
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input).to(self.configs.device)
                #mask_true_tensor = torch.FloatTensor(mask_true).to(self.configs.device)
                difference_x = input_tensor[:, 1:, :, :, :] - input_tensor[:, :-1, :, :, :]  # [8,4,128,160,3]
                next_differences, loss_p1 = self.pred_difference_frame_network(difference_x, self.configs.out_length)
                next_frames, loss_p2 = self.pred_network(input_tensor, self.configs.out_length, difference_x,
                                                         next_differences[0], next_differences[1])

                return next_frames.detach().cpu().numpy(), loss_p1.detach().cpu().numpy(), loss_p2.detach().cpu().numpy()

        else:
            input_tensor = torch.FloatTensor(input).to(self.configs.device)
            mask_true_tensor = torch.FloatTensor(mask_true).to(self.configs.device)
            next_frames, loss = self.network(input_tensor, mask_true_tensor)
            return next_frames.detach().cpu().numpy(), loss.detach().cpu().numpy()
