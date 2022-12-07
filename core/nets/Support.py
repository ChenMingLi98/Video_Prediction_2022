import matplotlib.pyplot as plt
import glob
import os
import torch
import numpy as np
import torchvision.utils as vutils
import cv2
import imageio
from termcolor import colored


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def save_training_samples(inp, save_path="examples"):
    # Reverse process before displaying
    inp = inp.cpu().numpy() * 255.0
    print("Input shape: {}".format(inp.shape))
    for i, video in enumerate(inp.squeeze(1)[:3]):
        path = os.path.join(save_path, "example_{}.gif".format(i))
        imageio.mimsave(path, video.astype(np.uint8), "GIF", fps=5)
        print("Saving example: {}".format(path))


def save_checkpoint(state, save_path="checkpoints/dualMotionGAN_0000.pth.tar"):
    print(colored("Saving checkpoints {}".format(save_path), color='green'))
    torch.save(state, save_path)


def showModelTraining(losses):
    plt.figure(figsize=(10, 5))
    plt.title("Model Loss During Training")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def latestCheckpoint(path, mask="*.pth.tar"):
    list_of_files = glob.glob(os.path.join(path, mask))
    latest_file = os.path.basename(max(list_of_files, key=os.path.getctime))
    print("Latest checkpoint: {}".format(latest_file))
    return latest_file


def collate(batch):
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)
    return batch[:, :, rand-10:rand], batch[:, :, rand]


# Visualize Test model results
def collate_test(batch):
    # Last 10 frames are target
    target = np.array(batch)[:, 10:]

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)
    batch = batch / 255.0
    batch = batch.to(device)
    return batch, target


def createPredImage(batch, output):
    # Prepare plot
    fig, axarr = plt.subplots(batch.shape[0], 11)
    fig.set_size_inches(20, 7)

    for i, sequence in enumerate(batch):
        for video in sequence:
            for j, frame in enumerate(video):
                if j < 10:
                    axarr[i, j].imshow(
                        np.transpose(vutils.make_grid(frame.to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
                    axarr[i, j].set_axis_off()
                    if i == 0:
                        axarr[i, j].title.set_text('Input {}'.format(j))

    print("Output: {}".format(output.shape))
    for i, sequence in enumerate(output):
        axarr[i, 10].imshow(np.transpose(vutils.make_grid(sequence[0].to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
        axarr[i, 10].set_axis_off()
        if i == 0:
            axarr[i, 10].title.set_text('Prediction')

    plt.show()


def save_output(save_dir, epoch, inp, target, prediction, flow_prediction, frame_prediction):
    batch_size, _, seq_len, height, width = inp.size()

    for i, sequence in enumerate(inp):
        # Prepare plot
        fig, axarr = plt.subplots(1, seq_len // 2 + 4)
        fig.set_size_inches(20, 7)
        for video in sequence:
            for j, frame in enumerate(video):
                if j >= seq_len//2:
                    axarr[j-seq_len//2].imshow(
                        np.transpose(vutils.make_grid(frame.to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
                    axarr[j-seq_len//2].set_axis_off()
                    axarr[j-seq_len//2].title.set_text('Input {}'.format(j))

            axarr[j+1-seq_len//2].imshow(
                np.transpose(vutils.make_grid(target[i].to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
            axarr[j+1-seq_len//2].set_axis_off()
            axarr[j+1-seq_len//2].set_title('Target', color="green", fontweight='bold')

            axarr[j+2-seq_len//2].imshow(
                np.transpose(vutils.make_grid(frame_prediction[i].to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
            axarr[j+2-seq_len//2].set_axis_off()
            axarr[j+2-seq_len//2].title.set_text('FramePred')

            prev = np.array(torch.squeeze(inp[:, :, -1][i].cpu()))
            flow = np.transpose(flow_prediction[i].cpu().detach().numpy(), (1, 2, 0))
            frame = torch.from_numpy(warp_flow(prev, flow))
            axarr[j+3-seq_len//2].imshow(
                np.transpose(vutils.make_grid(frame.to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
            axarr[j+3-seq_len//2].set_axis_off()
            axarr[j+3-seq_len//2].title.set_text('FlowWarp')

            axarr[j+4-seq_len//2].imshow(
                np.transpose(vutils.make_grid(prediction[i].to('cpu')[:64], padding=2, normalize=True), (1, 2, 0)))
            axarr[j+4-seq_len//2].set_axis_off()
            axarr[j+4-seq_len//2].set_title('Prediction', color="red", fontweight='bold')
            plt.savefig(os.path.join(save_dir, "epoch_{}_{}.png".format(epoch, i)))