import os
import json
import imageio
import cv2
import numpy as np
import torch
import torch.nn.functional as F


trans_t = lambda t : torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]
]).float()

rot_phi = lambda phi : torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]
]).float()

rot_theta = lambda theta : torch.Tensor([
    [np.cos(theta), 0, -np.sin(theta), 0],
    [0, 1, 0, 0],
    [np.sin(theta), 0, np.cos(theta), 0],
    [0, 0, 0, 1]
]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180. * np.pi) @ c2w
    c2w = rot_theta(theta/180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([
        [-1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0], 
        [0, 0, 0, 1]
    ])) @ c2w
    

def load_LINEMOD_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json', 'r')) as fp:
            metas[s] = json.load(fp)
        
    all_images = []
    all_poses = []
    counts = [0]

    for s in splits:
        meta = metas[s]
        images = []
        poses = []
        if s == 'train' or testskip == 0:
            skip = 1
        else: 
            skip = testskip
        
        for idx_test, frame in enumerate(meta['frames'][::skip]):
            fname = frame['file_path']
            if s == 'test':
                print(f"{idx_test}th test frame: {fname}")
            images.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        images = (np.array(images) / 255.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + images.shape[0])
        all_images.append(images)
        all_poses.append(poses)
        
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    images = np.concatenate(all_images, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = images[0].shape[:2]
    focal = float(meta['frames'][0]['intrinsic_matrix'][0][0])
    K = meta['frames'][0]['intrinsic_matrix']
    print(f"Focal: {focal}")
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)
    
    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2
        
        images_half_res = np.zeros((images.shape[0], H, W, 3))
        for i, image in enumerate(images):
            images_half_res[i] = cv2.resize(image, (H, W), interpolation=cv2.INTER_AREA)
        images = images_half_res
        
    near = np.floor(min(metas['train']['near'], metas['test']['near']))
    far = np.ceil(max(metas['train']['far'], metas['test']['far']))
    return images, poses, render_poses, [H, W, focal], K, i_split, near, far