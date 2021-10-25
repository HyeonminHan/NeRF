import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
from run_nerf_helpers import *
from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data, load_donerf_data
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import statistics

tf.compat.v1.enable_eager_execution()

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        embedded_dirs = embeddirs_fn(input_dirs_flat)

        embedded = tf.concat([embedded, embedded_dirs], -1)
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    def raw2outputs(raw, z_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract RGB of each sample position along each ray.
        rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to 
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        disp_map = 1./tf.maximum(1e-10, depth_map /
                                 tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]
    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    # print("rays_o in render_rays:", rays_o)
    # print("rays_d in render_rays:", rays_d)
    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]
    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)

    near2 = tf.fill(near.shape, 2.)
    far2 = tf.fill(far.shape, 6.)

    # print("near shape:", near)
    # print("far shape:", far)
    # print("depth:", (near+far)/2.)

    ''' GAUSSIAN SAMPLING
    z_vals = np.random.normal((near + far) / 2., 0.15, [near.shape[0], N_samples])
    z_vals = np.sort(z_vals, axis=1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    # print("gaussian z_vals:", z_vals)
    # print("s:", s)
    # print("s shape;", s.shape) ## 얘가 가우시안 깊이값 , 3차원 점의 좌표 생성 필요 
    # exit()
    '''


    '''SKEW DISTRIBUTION SAMPLING
    from scipy.stats import skewnorm

    skew = -2
    z_vals = skewnorm.rvs(skew, size=[near.shape[0], N_samples])
    z_vals = 0.15 * z_vals + (near + far) / 2.
    z_vals = np.sort(z_vals, axis=1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    # print("skew z_vals:" ,z_vals)
    # print("skew z_vals shape:" ,z_vals.shape)
    # exit()
    '''


    #'''#####################################################
    # near - 2.0, far - 6.0
    if not lindisp: ###here -lego
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
        z_vals2 = near2 * (1.-t_vals) + far2 * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])
    z_vals2 = tf.broadcast_to(z_vals2, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:  # default lego
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

        mids2 = .5 * (z_vals2[..., 1:] + z_vals2[..., :-1])
        upper2 = tf.concat([mids2, z_vals2[..., -1:]], -1)
        lower2 = tf.concat([z_vals2[..., :1], mids2], -1)
        # stratified samples in those intervals
        t_rand2 = tf.random.uniform(z_vals2.shape)
        z_vals2 = lower2 + (upper2 - lower2) * t_rand2

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]
    #'''######################################################



    
    '''

    z_vals2 = near2 * (1.-t_vals) + far2 * (t_vals)
    z_vals2 = tf.broadcast_to(z_vals2, [N_rays, N_samples])
    pts2 = rays_o[..., None, :] + rays_d[..., None, :] * \
    z_vals2[..., :, None]  # [N_rays, N_samples, 3]

    mids2 = .5 * (z_vals2[..., 1:] + z_vals2[..., :-1])
    upper2 = tf.concat([mids2, z_vals2[..., -1:]], -1)
    lower2 = tf.concat([z_vals2[..., :1], mids2], -1)
    # stratified samples in those intervals
    t_rand2 = tf.random.uniform(z_vals2.shape)
    z_vals2 = lower2 + (upper2 - lower2) * t_rand2
    
    global xs, ys, zs, xs_2, ys_2, zs_2,xs_3, ys_3, zs_3, origin_x, origin_y, origin_z
    global xs_,ys_, zs_, xs_3_,ys_3_, zs_3_
    # xs=[]
    # ys=[]
    # zs=[]

    xs = tf.concat([xs, tf.reshape(pts[...,0, 0], [-1])], 0)
    ys = tf.concat([ys, tf.reshape(pts[...,0, 1], [-1])], 0)
    zs = tf.concat([zs, tf.reshape(pts[...,0, 2], [-1])], 0)

    xs_2 = tf.concat([xs_2, tf.reshape(pts[...,int(N_samples/2), 0], [-1])], 0)
    ys_2 = tf.concat([ys_2, tf.reshape(pts[...,int(N_samples/2), 1], [-1])], 0)
    zs_2 = tf.concat([zs_2, tf.reshape(pts[...,int(N_samples/2), 2], [-1])], 0)

    xs_3 = tf.concat([xs_3, tf.reshape(pts[...,-1, 0], [-1])], 0)
    ys_3 = tf.concat([ys_3, tf.reshape(pts[...,-1, 1], [-1])], 0)
    zs_3 = tf.concat([zs_3, tf.reshape(pts[...,-1, 2], [-1])], 0)

    xs_ = tf.concat([xs_, tf.reshape(pts2[...,0, 0], [-1])], 0)
    ys_ = tf.concat([ys_, tf.reshape(pts2[...,0, 1], [-1])], 0)
    zs_ = tf.concat([zs_, tf.reshape(pts2[...,0, 2], [-1])], 0)

    xs_3_ = tf.concat([xs_3_, tf.reshape(pts2[...,-1, 0], [-1])], 0)
    ys_3_ = tf.concat([ys_3_, tf.reshape(pts2[...,-1, 1], [-1])], 0)
    zs_3_ = tf.concat([zs_3_, tf.reshape(pts2[...,-1, 2], [-1])], 0)

    origin_x = tf.concat([origin_x, tf.reshape(rays_o[0][0], [-1])], 0)
    origin_y = tf.concat([origin_y, tf.reshape(rays_o[0][1], [-1])], 0)
    origin_z = tf.concat([origin_z, tf.reshape(rays_o[0][2], [-1])], 0)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(xs_b, ys_b, zs_b, c='r', marker='o', s=[0.1 for x in range(xs_b.shape[0])])
    ax.scatter(xs, ys, zs, c='y', marker='o', s=[
               0.1 for x in range(xs.shape[0])])
    ax.scatter(xs_2, ys_2, zs_2, c='b', marker='o', s=[
        0.1 for x in range(xs_2.shape[0])])
    ax.scatter(xs_3, ys_3, zs_3, c='y', marker='o', s=[
        0.1 for x in range(xs_3.shape[0])])
    ax.scatter(xs_, ys_, zs_, c='g', marker='o', s=[
        0.1 for x in range(xs_.shape[0])])
    ax.scatter(xs_3_, ys_3_, zs_3_, c='g', marker='o', s=[
        0.1 for x in range(xs_3_.shape[0])])
    ax.scatter(origin_x, origin_y, origin_z, c='r', marker='o', s=[3.0])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    '''

    # Evaluate model at each point.
    raw = network_query_fn(pts, viewdirs, network_fn)  # [N_rays, N_samples, 4]
    
    # for layer in network_fn.layers :
    #     print("layer===",layer.output_shape)
    #     if len(layer.get_weights())>0 :

    #         # print("weight:", np.array(layer.get_weights()).shape)
    #         print("weight[0]:", len(layer.get_weights()[0]))
    #         print("weight[0][0]:", len(layer.get_weights()[0][0]))

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)


        pts_coarse = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples, 3]

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        '''
        pts_passed = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_samples[..., :, None]  # [N_rays, N_importance, 3]

        global xs, ys, zs, xs_2, ys_2, zs_2,xs_3, ys_3, zs_3, origin_x, origin_y, origin_z
        global xs_,ys_, zs_, xs_3_,ys_3_, zs_3_
        # xs=[]
        # ys=[]
        # zs=[]


        xs_c = tf.concat([xs, tf.reshape(pts_coarse[..., 0], [-1])], 0)
        ys_c = tf.concat([ys, tf.reshape(pts_coarse[..., 1], [-1])], 0)
        zs_c = tf.concat([zs, tf.reshape(pts_coarse[..., 2], [-1])], 0)

        xs = tf.concat([xs, tf.reshape(pts_passed[..., 0], [-1])], 0)
        ys = tf.concat([ys, tf.reshape(pts_passed[..., 1], [-1])], 0)
        zs = tf.concat([zs, tf.reshape(pts_passed[..., 2], [-1])], 0)

        # xs_2 = tf.concat([xs_2, tf.reshape(pts[...,int(N_samples/2), 0], [-1])], 0)
        # ys_2 = tf.concat([ys_2, tf.reshape(pts[...,int(N_samples/2), 1], [-1])], 0)
        # zs_2 = tf.concat([zs_2, tf.reshape(pts[...,int(N_samples/2), 2], [-1])], 0)

        # xs_3 = tf.concat([xs_3, tf.reshape(pts[...,-1, 0], [-1])], 0)
        # ys_3 = tf.concat([ys_3, tf.reshape(pts[...,-1, 1], [-1])], 0)
        # zs_3 = tf.concat([zs_3, tf.reshape(pts[...,-1, 2], [-1])], 0)


        origin_x = tf.concat([origin_x, tf.reshape(rays_o[0][0], [-1])], 0)
        origin_y = tf.concat([origin_y, tf.reshape(rays_o[0][1], [-1])], 0)
        origin_z = tf.concat([origin_z, tf.reshape(rays_o[0][2], [-1])], 0)


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ax.scatter(xs_b, ys_b, zs_b, c='r', marker='o', s=[0.1 for x in range(xs_b.shape[0])])
        # ax.scatter(xs_c, ys_c, zs_c, c='b', marker='o', s=[
        #         0.1 for x in range(xs.shape[0])])
        ax.scatter(xs, ys, zs, c='y', marker='o', s=[
                0.1 for x in range(xs.shape[0])])
        # ax.scatter(xs_2, ys_2, zs_2, c='b', marker='o', s=[
            # 0.1 for x in range(xs_2.shape[0])])
        # ax.scatter(xs_3, ys_3, zs_3, c='y', marker='o', s=[
            # 0.1 for x in range(xs.shape[0])])
        # ax.scatter(xs_, ys_, zs_, c='g', marker='o', s=[
            # 0.1 for x in range(xs_.shape[0])])
        # ax.scatter(xs_3_, ys_3_, zs_3_, c='g', marker='o', s=[
            # 0.1 for x in range(xs.shape[0])])
        ax.scatter(origin_x, origin_y, origin_z, c='r', marker='o', s=[3.0])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
        '''


        # Make predictions with network_fine.
        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None, depth_img = None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d, depth = get_rays(H, W, focal, c2w, depth_img=depth_img)
    else:
        # use provided ray batch
        if depth_img is not None:
            rays_o, rays_d, depth = rays
        else :
            rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        # print("use_viewdirs")

        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d, depth = get_rays(H, W, focal, c2w_staticcam, depth_img=depth_img)
            print("c2w_staticcam")

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32) #[ray batch, 3]
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32) #[ray batch, 3]
    
    #if depth_img is not None:
    #    rays_o = rays_o[depth[...,0]>0]
    #    rays_d = rays_d[depth[...,0]>0]

    if depth_img is not None:
        global args 

        if args.dataset_type == 'blender':
            quantiz_coef = 8.
            depth = (1. - tf.reshape(depth[...,0], [-1,1])) * quantiz_coef #* 4.0 + 2.0 #* 6.0 

            if use_backgd :

                near = tf.where(depth < quantiz_coef, depth - args.alpha, tf.constant(args.near,shape=depth.shape))
                far = tf.where(depth < quantiz_coef, depth + args.alpha, tf.constant(args.far,shape=depth.shape))

            else :
                near = depth - args.alpha
                far = depth + args.alpha

        elif args.dataset_type == 'donerf':

            depth = (tf.reshape(depth[...,0], [-1,1])) 
            alpha = (args.far -args.near)*args.alpha / 8. #blender data에서 전체 길이 8일때 alpha 0.5 등의 scale을 맞춰주기 위함
            # print("alpha:", alpha)
            if use_backgd :
                # print("near1:", near)
                # print("far1:", far)
                near = tf.where(depth > 0, depth - alpha, tf.constant(args.near,shape=depth.shape))
                far = tf.where(depth > 0, depth + alpha, tf.constant(args.far,shape=depth.shape))
                # print("near2:", near)
                # print("far2:", far)
            else :
                near = depth - args.alpha
                far = depth + args.alpha

        
        ####################################
        # rays_o [N_rays,3]
        # rays_d [N_rays,3]
        # depth [N_rays,1]
        
        '''
        print("rays_o", rays_o)
        print("rays_d", rays_d)
        print("depth", depth)

        flag_foregd = depth < 8
        flag_backgd = depth >= 8

        fore_mask = np.column_stack([flag_foregd,flag_foregd,flag_foregd])
        back_mask = np.column_stack([flag_backgd,flag_backgd,flag_backgd])




        pts_f = np.reshape(rays_o[fore_mask],(-1,3))[..., None, :] + np.reshape(rays_d[fore_mask],(-1,3))[..., None, :] * \
            depth[flag_foregd][..., None, None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]

        pts_b = np.reshape(rays_o[back_mask],(-1,3))[..., None, :] + np.reshape(rays_d[back_mask],(-1,3))[..., None, :] * \
            depth[flag_backgd][..., None, None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]

        print("pts_b", pts_b.shape)
        # np.set_printoptions(threshold=sys.maxsize)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        depth[..., :, None]

        global xs, ys, zs, origin_x, origin_y, origin_z, xs_2, ys_2, zs_2, xs_3, ys_3, zs_3

        xs=[]
        ys=[]
        zs=[]

        xs = tf.concat([xs, tf.reshape(pts[...,0], [-1])], 0)
        ys = tf.concat([ys, tf.reshape(pts[...,1], [-1])], 0)
        zs = tf.concat([zs, tf.reshape(pts[...,2], [-1])], 0)

        origin_x = tf.concat([origin_x, tf.reshape(rays_o[0][0], [-1])], 0)
        origin_y = tf.concat([origin_y, tf.reshape(rays_o[0][1], [-1])], 0)
        origin_z = tf.concat([origin_z, tf.reshape(rays_o[0][2], [-1])], 0)

        xs_b = tf.reshape(pts_b[...,0], [-1]) 
        ys_b = tf.reshape(pts_b[...,1], [-1])
        zs_b = tf.reshape(pts_b[...,2], [-1])
        
        print("len(xs)", len(xs))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print("len(xs_b)", len(xs_b))

        near_ = rays_o[..., None, :] + rays_d[..., None, :] * 2.
        far_ = rays_o[..., None, :] + rays_d[..., None, :] * 6.

        # pts2 = rays_o[..., None, :] + rays_d[..., None, :] * \
        #z_vals2[..., :, None]  # [N_rays, N_samples, 3]
    

        #pts_near = np.reshape(rays_o,(-1,3))[..., None, :] + np.reshape(rays_d,(-1,3))[..., None, :] * \
        #    near[..., None, None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]

        # pts_far = np.reshape(rays_o,(-1,3))[..., None, :] + np.reshape(rays_d,(-1,3))[..., None, :] * \
        #     far[..., None, None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]

        pts_near = rays_o[..., None, :] + rays_d[..., None, :] * \
           near[...,  None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]

        pts_far = rays_o[..., None, :] + rays_d[..., None, :] * \
            far[..., None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]


        print("pts_near:", pts_near.shape)
        print("pts_far:", pts_far.shape)

        pts_f_n = np.reshape(rays_o[fore_mask],(-1,3))[..., None, :] + np.reshape(rays_d[fore_mask],(-1,3))[..., None, :] * \
            near[flag_foregd][..., None, None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]

        pts_f_f = np.reshape(rays_o[fore_mask],(-1,3))[..., None, :] + np.reshape(rays_d[fore_mask],(-1,3))[..., None, :] * \
            far[flag_foregd][..., None, None]  # [N_rays, N_samples, 3] -> using depth [N_rays, N_samples=1, 3]


        print("pts_f_n:", pts_f_n.shape)
        print("pts_f_f:", pts_f_f.shape)
        xs_temp_n = tf.reshape(near_[...,0], [-1])
        ys_temp_n = tf.reshape(near_[...,1], [-1])
        zs_temp_n = tf.reshape(near_[...,2], [-1])
        xs_temp_f = tf.reshape(far_[...,0], [-1]) 
        ys_temp_f = tf.reshape(far_[...,1], [-1])
        zs_temp_f = tf.reshape(far_[...,2], [-1])

        xs = tf.concat([xs, tf.reshape(pts[...,0], [-1])], 0)
        ys = tf.concat([ys, tf.reshape(pts[...,1], [-1])], 0)
        zs = tf.concat([zs, tf.reshape(pts[...,2], [-1])], 0)


        xs_2 = tf.concat([xs_2, tf.reshape(pts_near[...,0], [-1])], 0) 
        ys_2 = tf.concat([ys_2, tf.reshape(pts_near[...,1], [-1])], 0) 
        zs_2 = tf.concat([zs_2, tf.reshape(pts_near[...,2], [-1])], 0) 
        xs_3 = tf.concat([xs_3, tf.reshape(pts_far[...,0], [-1])], 0) 
        ys_3 = tf.concat([ys_3, tf.reshape(pts_far[...,1], [-1])], 0) 
        zs_3 = tf.concat([zs_3, tf.reshape(pts_far[...,2], [-1])], 0) 

        origin_x = tf.concat([origin_x, tf.reshape(rays_o[0][0], [-1])], 0)
        origin_y = tf.concat([origin_y, tf.reshape(rays_o[0][1], [-1])], 0)
        origin_z = tf.concat([origin_z, tf.reshape(rays_o[0][2], [-1])], 0)

        xs_f_n = tf.reshape(pts_f_n[...,0], [-1]) 
        ys_f_n = tf.reshape(pts_f_n[...,1], [-1])
        zs_f_n = tf.reshape(pts_f_n[...,2], [-1])
        xs_f_f = tf.reshape(pts_f_f[...,0], [-1])
        ys_f_f = tf.reshape(pts_f_f[...,1], [-1])
        zs_f_f = tf.reshape(pts_f_f[...,2], [-1])

        # ax.scatter(xs_b, ys_b, zs_b, c='r', marker='o', s=[0.1 for x in range(xs_b.shape[0])])
        ax.scatter(xs, ys, zs, c='b', marker='o', s=[0.1 for x in range(xs.shape[0])])
        ax.scatter(origin_x, origin_y, origin_z, c='r', marker='o', s=[3.0])
        # ax.scatter(xs_temp_n, ys_temp_n, zs_temp_n, c='g', marker='o', s=[0.1 for x in range(xs_temp_n.shape[0])])
        # ax.scatter(xs_temp_f, ys_temp_f, zs_temp_f, c='g', marker='o', s=[0.1 for x in range(xs_temp_f.shape[0])])

        ax.scatter(xs_2, ys_2, zs_2, c='y', marker='o', s=[0.1 for x in range(xs_2.shape[0])])
        ax.scatter(xs_3, ys_3, zs_3, c='y', marker='o', s=[0.1 for x in range(xs_3.shape[0])])

        # ax.scatter(xs_f_n, ys_f_n, zs_f_n, c='y', marker='o', s=[0.1 for x in range(xs_f_n.shape[0])])
        # ax.scatter(xs_f_f, ys_f_f, zs_f_f, c='y', marker='o', s=[0.1 for x in range(xs_f_f.shape[0])])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        #ax.set_xlim([-2,1])
        #ax.set_ylim([-1,4])
        #ax.set_zlim([-0.5,3])

        plt.show()
        #return None, None, None, None
        '''

    else : 
       near, far = near * \
           tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])

    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print("nea:", near)
    # print("far:", far)

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)
    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)
        

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0, depth_imgs=None):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    losses = []
    psnrs = []
    times = []

    t = time.time()
    
    for i, c2w in enumerate(render_poses):
        print("i :", i, "time:", time.time() - t)
        times.append(time.time() - t)

        t = time.time()
        if depth_imgs is not None:
            rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], depth_img=depth_imgs[i], **render_kwargs)

            if not use_backgd : 
                depth = depth_imgs[i][...,:3]
                rgb = tf.where(depth > 0, rgb, tf.ones(depth.shape))
        else : 
            rgb, disp, acc, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor == 0:
            if depth_imgs is not None :
                if use_backgd :
                    l = np.mean(np.square(rgb - gt_imgs[i]))
                    p = -10. * np.log10(l)

                else :
                    l = np.mean(np.square(rgb[depth>0] - gt_imgs[i][depth>0]))
                    p = -10. * np.log10(l)

            else : 
                if not cal_backgd :
                    l = np.mean(np.square(rgb[gt_imgs[i]<1.] - gt_imgs[i][gt_imgs[i]<1.]))
                    p = -10. * np.log10(l)
                else:
                    l = np.mean(np.square(rgb - gt_imgs[i]))
                    p = -10. * np.log10(l)

            print("loss : ", l)
            print("psnr : ", p)
            losses.append(l)
            psnrs.append(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps, losses, psnrs, times

def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(
                ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='/media/hyunmin/FED62A69D62A21FF/nerf/logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--use_depth", action='store_true',
                        help='use depth information in training')
    parser.add_argument("--alpha", type=float, default=0.5,
                        help='sampling range in depth based sampling')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')    

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / donerf')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--near", type=float, default=0.,
                        help='near value of dataset')
    parser.add_argument("--far", type=float, default=1.,
                        help='far value of dataset')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=1000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=1000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=100000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=10000000,
                        help='frequency of render_poses video saving')

    return parser


xs=[]
ys=[]
zs=[]
xs_2=[]
ys_2=[]
zs_2=[]
xs_3=[]
ys_3=[]
zs_3=[]
xs_=[]
ys_=[]
zs_=[]
xs_2_=[]
ys_2_=[]
zs_2_=[]
xs_3_=[]
ys_3_=[]
zs_3_=[]

origin_x=[]
origin_y=[]
origin_z=[]

args = None
use_backgd = False #depth based
cal_backgd = False #original nerf


def train():

    parser = config_parser()
    global args 
    args = parser.parse_args()
    
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data
    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        if args.use_depth : 
            images, poses, render_poses, hwf, i_split, depth_imgs = load_blender_data(
            args.datadir, args.half_res, args.testskip, args.use_depth)
        else:
            images, poses, render_poses, hwf, i_split = load_blender_data(
                args.datadir, args.half_res, args.testskip, args.use_depth)
        print('Loaded blender', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.
        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]
        if args.use_depth:
            render_depths = np.array(depth_imgs[i_test])
        else:
            render_depths = None
        
    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'donerf':
        if args.use_depth : 
            images, poses, render_poses, hwf, i_split, depth_imgs = load_donerf_data(
            args.datadir, args.half_res, args.testskip, args.use_depth)
        else:
            images, poses, render_poses, hwf, i_split = load_donerf_data(
                args.datadir, args.half_res, args.testskip, args.use_depth)
        print('Loaded donerf', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        # print("i_train:", i_train)
        # print("i_val:", i_val)
        # print("i_test:", i_test)

        if args.config == 'config_pavillon.txt':
            near = 1.1224385499954224
            far = 118.75775413513185

            if args.use_depth:
                for i in range(len(depth_imgs)):
                    # print("depth_img before:", depth_imgs[i])
                    depth_imgs[i] = np.where(depth_imgs[i]==1e+10, far, depth_imgs[i])
                    # print("depth_img after:", depth_imgs[i])

        elif args.config == 'config_lego_donerf.txt':
            near = 0.5999020725488663
            far = 3.6212123036384583

            if args.use_depth:
                for i in range(len(depth_imgs)):
                    # print("depth_img before:", depth_img)
                    depth_imgs[i] = np.where(depth_imgs[i]==10, 0, depth_imgs[i])
                    # print("depth_img after:", depth_img)
        else :
            print("pls insert near, far values")
            exit()
        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]
        if args.use_depth:
            render_depths = np.array(depth_imgs[i_test])
        else:
            render_depths = None

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    print("=============focal:", focal)

    print("#########i_test", i_test)
    if args.render_test:
        render_poses = np.array(poses[i_test])
        print("render_poses", render_poses)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)
    
    iters_train=[]
    losses_train=[]
    psnrs_train=[]
    losses_val=[]
    psnrs_val=[]
    avr_dt = 0
    
    if start > 0  :
        plotimgdir = os.path.join(basedir, expname, 'plot_imgs')
        if os.path.isfile(plotimgdir+"/train_val_log.txt") :
            with open(plotimgdir+"/train_val_log.txt",'r') as f:
                lines = f.readlines()
                for l in lines:
                    x = l.split()
                    iters_train.append(int(x[1]))
                    losses_train.append(float(x[9]))
                    psnrs_train.append(float(x[11]))
                    losses_val.append(float(x[15]))
                    psnrs_val.append(float(x[17]))
                avr_dt = float(lines[-1].split()[5])
    
    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None


        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        rgbs, _,_,_,_ = render_path(render_poses, hwf, args.chunk, render_kwargs_test,
                              gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor, depth_imgs=render_depths)
        print('Done rendering', testsavedir)
        imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),
                         to8b(rgbs), fps=30, quality=8)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    N_iters = 400001
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()
        depth_img = None

        # Sample random ray batch

        if use_batching:
            print("use_batching")
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        else:   #default lego

            # Random from one image
            img_i = np.random.choice(i_train)
            target = images[img_i]
            pose = poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d, _ = get_rays(H, W, focal, pose)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                
                elif args.use_depth :
                    if use_backgd : 
                        depth_img = depth_imgs[img_i] 
                        coords = tf.stack(tf.meshgrid(
                            tf.range(H), tf.range(W), indexing='ij'), -1) #[H, W, 2]
                        
                    else :
                        depth_img = depth_imgs[img_i]
                        ii, jj = np.where(depth_img[...,0]>0)

                        coords = tf.stack([ii, jj], -1)

                
                else : ## lego default
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1) #[H, W, 2]
                coords = tf.reshape(coords, [-1, 2]) # [HxW, 2]

                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)

                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis]) # [ray_batch , 2]
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)

                if args.use_depth:
                    depth_img = tf.gather_nd(depth_img, select_inds)
                    batch_rays = tf.stack([rays_o, rays_d, depth_img[...,:3]], 0)

                else :                
                    batch_rays = tf.stack([rays_o, rays_d], 0)

                
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            if batch_rays is None :
                print("batch_rays is none!!!")
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays,
                verbose=i < 10, retraw=True, depth_img=depth_img, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            if not cal_backgd:
                loss_train = np.mean(np.square(rgb[target_s<1.] - target_s[target_s<1.]))
                psnr_train = -10. * np.log10(loss_train)

            else:
                loss_train = img_loss
                psnr_train = psnr


            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:

            rgbs, disps,_,_ = render_path(
                render_poses, hwf, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4',
                             to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4',
                             to8b(disps / np.max(disps)), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                rgbs_still, _,_,_,_ = render_path(
                    render_poses, hwf, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                imageio.mimwrite(moviebase + 'rgb_still.mp4',
                                 to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            n = 100

            # For each set of style and range settings, plot n random points in the box
            # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
            #for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:

            _,_, losses, psnrs, times = render_path(poses[i_test], hwf, args.chunk, render_kwargs_test,
                        gt_imgs=images[i_test], savedir=testsavedir, depth_imgs=render_depths)
            
            avr_loss=0
            avr_psnr=0
            avr_time=0
            f = open(testsavedir+"/testlog_"+str(i)+".txt", "a")
            for ii in range(len(i_test)):

                f.write('iter: {} one_render_time: {:.05f} test_img_i: {} test_loss: {:.7f} test_psnr: {:.4f}\n'\
                                    .format(i, times[ii], i_test[ii],losses[ii],psnrs[ii]))

                avr_loss += losses[ii]
                avr_psnr += psnrs[ii]
                avr_time += times[ii]

            avr_loss /= len(i_test)            
            avr_psnr /= len(i_test)             
            avr_time /= len(i_test)  

            f.write('iter: {} avr_train_time: {:.05f} avr_render_time: {:.05f} avr_loss: {:.7f} avr_psnr: {:.4f} stddev_of_psnrs: {:.4f}\n'\
                                    .format(i, avr_dt, avr_time, avr_loss, avr_psnr, statistics.stdev(psnrs)))
            f.close()


            print('Saved test set')

        if i % args.i_print == 0 or i < 10:
            print(args.config)
            print(expname, i, psnr_train, loss_train, global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('psnr0', psnr0)
            
            if i % args.i_img == 0:

                # Log a rendered validation view to Tensorboard
                img_val_i = np.random.choice(i_val)
                target = images[img_val_i]
                pose = poses[img_val_i, :3, :4]


                if args.use_depth:
                    rgb, disp, acc, extras = render(
                        H, W, focal, chunk=args.chunk, c2w=pose, depth_img=depth_imgs[img_val_i], **render_kwargs_test)

                    if use_backgd : 
                        
                        loss_val = img2mse(rgb, target)
                        psnr_val = mse2psnr(loss_val)

                    else :
                        depth = depth_imgs[img_val_i][...,:3]
                        rgb = tf.where(depth > 0, rgb, tf.ones(depth.shape))
                        
                        loss_val = np.mean(np.square(rgb[depth>0] - target[depth>0]))
                        psnr_val = -10. * np.log10(loss_val)

                else:
                    rgb, disp, acc, extras = render(H, W, focal, chunk=args.chunk, c2w=pose, **render_kwargs_test)

                    if not cal_backgd:
                        loss_val = np.mean(np.square(rgb[target<1.] - target[target<1.]))
                        psnr_val = -10. * np.log10(loss_val)
                    else :
                        loss_val = img2mse(rgb, target)
                        psnr_val = mse2psnr(loss_val)

                print("train loss:", loss)
                print("train loss_train:", loss_train)
                print("train psnr:", psnr_train)
                print("val loss:", loss_val)
                print("val psnr:", psnr_val)
                
                # Save out the validation image for Tensorboard-free monitoring

                if len(iters_train) == 0 or i > iters_train[-1]  : 
                    testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                    os.makedirs(testimgdir, exist_ok=True)
                    imageio.imwrite(os.path.join(testimgdir, '{:06d}.png'.format(i)), to8b(rgb))

                    iters_train.append(i)
                    losses_train.append(loss_train)
                    psnrs_train.append(psnr_train)
                    losses_val.append(loss_val)
                    psnrs_val.append(psnr_val)

                    interval_train = range(len(iters_train))

                    plotimgdir = os.path.join(basedir, expname, 'plot_imgs')
                    os.makedirs(plotimgdir, exist_ok=True)

                    plt.figure(1, figsize=(10, 5))
                    plt.title("Training Loss")
                    line1, = plt.plot(losses_train, 'b', label="train")
                    line2, = plt.plot(losses_val, 'r', label="val")
                    plt.xticks(interval_train, iters_train)
                    plt.ylim([0., 0.02])
                    plt.xlabel("Iteration")
                    plt.ylabel("Loss")
                    plt.legend(handles=(line1, line2),bbox_to_anchor=(1, 1.15), ncol=2)
                    plt.savefig(plotimgdir+'/loss_'+str(i)+'.png')

                    plt.figure(2, figsize=(10, 5))
                    plt.title("Training psnr")
                    line1, = plt.plot(psnrs_train, 'b', label="train")
                    line2, = plt.plot(psnrs_val, 'r', label="val")
                    plt.xticks(interval_train, iters_train)
                    plt.ylim([10, 33])
                    plt.xlabel("Iteration")
                    plt.ylabel("PSNR")
                    plt.legend(handles=(line1, line2),bbox_to_anchor=(1, 1.15), ncol=2)
                    plt.savefig(plotimgdir+'/psnr_'+str(i)+'.png')
                    #plt.show()

                    N = i / args.i_img
                    avr_dt = avr_dt * N / (N+1) + dt / (N+1)

                    f = open(plotimgdir+"/train_val_log.txt", "a")
                    f.write('iter: {} one_iter_time: {:.05f} avr_iter_time: {:.05f} train_img_i: {} train_loss: {:.7f} train_psnr: {:.4f} \
                                    val_img_i: {} val_loss: {:.7f} val_psnr: {:.4f}\n'\
                                        .format(i, dt, avr_dt,img_i,loss_train,psnr_train,\
                                                img_val_i, loss_val, psnr_val))

                    f.close()

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image(
                        'disp', disp[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image(
                        'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr_val)
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image(
                            'rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        tf.contrib.summary.image(
                            'disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image(
                            'z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()
