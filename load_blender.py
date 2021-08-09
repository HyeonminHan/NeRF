import os
import tensorflow as tf
import numpy as np
import imageio 
import json




trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def load_blender_data(basedir, half_res=False, testskip=1, use_depth=False):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depths = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        depth_imgs  = []

        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            if use_depth :
                fname_ = fname.split("/")
                fname_d = '/'.join(fname_[:-1]) + '/' + fname_[-1][:-4] + "_depth_0001.png"
                depth_imgs.append(imageio.imread(fname_d))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        if use_depth :
            depth_imgs = (np.array(depth_imgs) / 255.).astype(np.float32) 
            all_depths.append(depth_imgs)



    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    if use_depth :
        depth_imgs = np.concatenate(all_depths, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    print()
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.
    # print("imgs", imgs.shape)
    # print("imgs", depth_imgs.shape)
    # print("i_split", i_split)

    import matplotlib.pyplot as plt

    if use_depth:
        return imgs, poses, render_poses, [H, W, focal], i_split, depth_imgs
    else :
        return imgs, poses, render_poses, [H, W, focal], i_split


