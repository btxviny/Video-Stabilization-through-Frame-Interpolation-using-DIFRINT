import numpy as np
import cv2
import argparse
import os
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from models import ResNet, UNet
from PWC_src import PWC_Net, flow_to_image
from PWC_src.pwc import FlowEstimate
H,W = 256,256
device = 'cuda'

class DIFRINT(nn.Module):
    def __init__(self):
        super(DIFRINT,self).__init__()
        self.resnet = ResNet(hidden_size=64).to(device).eval()
        self.unet = UNet(hidden_size=64).to(device).eval()
        self.pwc = PWC_Net('./ckpt/sintel.pytorch')
        self.pwc.to(device).eval()

    def get_flow(self,img1,img2):
        img1_t = (img1 + 1) / 2 
        img2_t = (img2 + 1) / 2 
        flow = FlowEstimate(img1_t,img2_t, self.pwc)
        return flow.detach()
    
    def forward(self, ft_minus, ft, ft_plus):
        flo1 = self.get_flow(ft_minus, ft)
        flo2 = self.get_flow(ft_plus, ft)
        warped1 = dense_warp(ft_minus,0.5 * flo1)
        warped2 = dense_warp(ft_plus,0.5 * flo2)
        fint = self.unet(warped1, warped2, flo1, flo2, ft_minus, ft_plus)
        flo3 = self.get_flow(ft,fint)
        warped3 = dense_warp(ft,flo3)
        fout = self.resnet(fint, warped3)
        return fint, fout

def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using CAIN')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()

def save_video(frames, path):
    frame_count,h,w,_ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w,h))
    for idx in range(frame_count):
        out.write(frames[idx,...])
    out.release()

def dense_warp(image, flow):
    """
    Densely warps an image using optical flow.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
        flow (torch.Tensor): Optical flow tensor of shape (batch_size, 2, height, width).

    Returns:
        torch.Tensor: Warped image tensor of shape (batch_size, channels, height, width).
    """
    batch_size, channels, height, width = image.size()

    # Generate a grid of pixel coordinates based on the optical flow
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width),indexing ='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).to(image.device)
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
    new_grid = grid + flow.permute(0, 2, 3, 1)

    # Normalize the grid coordinates between -1 and 1
    new_grid /= torch.tensor([width - 1, height - 1], dtype=torch.float32, device=image.device)
    new_grid = new_grid * 2 - 1
    # Perform the dense warp using grid_sample
    warped_image = F.grid_sample(image, new_grid, align_corners=False)

    return warped_image
    
def stabilize(in_path,out_path):
    
    if not os.path.exists(in_path):
        print(f"The input file '{in_path}' does not exist.")
        exit()
    _,ext = os.path.splitext(in_path)
    if ext not in ['.mp4','.avi']:
        print(f"The input file '{in_path}' is not a supported video file (only .mp4 and .avi are supported).")
        exit()

    #Load frames and stardardize
    cap = cv2.VideoCapture(in_path)
    frames = []
    while True:
        ret,img = cap.read()
        if not ret:
            break
        img = ((img / 255.0) * 2) - 1 
        frames.append(img)
    frames = np.array(frames, dtype = np.float32)
    frame_count = frames.shape[0]
    
    # stabilize video
    SKIP = 1
    ITER = 3
    interpolated = frames.copy()
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    for iter in range(ITER):
        print(iter)
        temp = interpolated.copy()
        for frame_idx in range(SKIP,frame_count - SKIP):
            torch.cuda.empty_cache()
            ft_minus = torch.from_numpy(interpolated[frame_idx - SKIP,...]).permute(2,0,1).unsqueeze(0).to(device)
            ft = torch.from_numpy(frames[frame_idx]).permute(2,0,1).unsqueeze(0).to(device)
            ft_plus = torch.from_numpy(interpolated[frame_idx + SKIP,...]).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad(): 
                fint,fout = difrint(ft_minus,ft,ft_plus)
            temp[frame_idx,...] = fout.cpu().squeeze(0).permute(1,2,0).numpy()
            img  = (((fout.cpu().squeeze(0).permute(1,2,0).numpy() + 1) / 2)*255.0).astype(np.uint8)
            cv2.imshow('window',img)
            if cv2.waitKey(1) & 0xFF == ord('9'):
                break
        interpolated = temp.copy()
    cv2.destroyAllWindows()
    stable_frames = np.clip((255 *(interpolated + 1) / 2),0,255).astype(np.uint8)
    save_video(stable_frames,out_path)

if __name__ == '__main__':
    args = parse_args()
    difrint = DIFRINT().eval().to(device)
    unet_path = './ckpts/unet/'
    ckpts = os.listdir(unet_path)
    if ckpts:
        ckpts = sorted(ckpts, key=lambda x: int(x.split('.')[0].split('_')[1]))
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(unet_path, latest))
        difrint.unet.load_state_dict(state_dict['model'])
        print(f'Loaded pretrained UNet {latest}')
    resnet_path = './ckpts/resnet/'
    ckpts = os.listdir(resnet_path)
    if ckpts:
        ckpts = sorted(ckpts, key=lambda x: int(x.split('.')[0].split('_')[1]))
        latest = ckpts[-1]
        state_dict = torch.load(os.path.join(resnet_path, latest))
        difrint.resnet.load_state_dict(state_dict['model'])
        print(f'Loaded pretrained ResNet {latest}')
    # Load ResNet checkpoints
    
    stabilize(args.in_path, args.out_path)