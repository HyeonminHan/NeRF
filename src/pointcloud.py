import numpy as np
import open3d
import cv2
import tensorflow as tf
import imageio
import matplotlib.pyplot as plt
import statistics

def make_PC(color_imgs, depth_imgs, f, Rts, pc_viewnum, back_value):

    H = color_imgs[0].shape[0]
    W = color_imgs[0].shape[1]
    K = np.array([[f, 0, W/2.], [0, f, H/2.], [0, 0, 1]])

    depth_threshold = 0.1

    ## 첫번째 영상으로 PC 생성
    xyz_worldCoor, rgb_colors = reconstruction_2dto3d(color_imgs[0], depth_imgs[0], f, Rts[0], H, W) # 800x800x3

    ### 배경 제외 생성 
    # EXR 
    # back_value is pixel value of background
    # if you do not want to remove background, you should change 'back_value' as '-1'
    ###
    xyz_worldCoor = xyz_worldCoor[np.where(np.logical_and(depth_imgs[0][...,0]< 1000, depth_imgs[0][...,0] != back_value))]
    rgb_colors = rgb_colors[np.where(np.logical_and(depth_imgs[0][...,0]< 1000, depth_imgs[0][...,0] != back_value))]


    # mode = 0 # ALL PC
    mode = 1 # INCREMENTAL PC

    # ALL PC
    if mode == 0:
        for i in range(1, 2, 5):#(1, 100, 20):

            xyz_worldCoor_i, rgb_colors_i = reconstruction_2dto3d(color_imgs[i], depth_imgs[i], f, Rts[i], H, W) # 800x800x3
            k = np.where(np.logical_and(depth_imgs[0][...,0]< 1000,depth_imgs[0][...,0] != back_value))
            xyz_worldCoor = np.concatenate([xyz_worldCoor, xyz_worldCoor_i[k]], axis=0)
            rgb_colors = np.concatenate([rgb_colors, rgb_colors_i[k]], axis=0)
            print("xyz_worldCoor", xyz_worldCoor.shape)

            # d = depth_imgs[i][...,:3]
            # loss = np.mean(np.square(color_mat[d<1000] - color_imgs[i][...,:3][d<1000]))
            # psnr = -10. * np.log10(loss)
            # print("psnr:", psnr)

            # plt.figure()
            # plt.imshow(dist_mat)

            # plt.figure()
            # plt.imshow(color_mat)

            # plt.figure()
            # plt.imshow(color_imgs[i])
            # plt.show()
            
        source = open3d.geometry.PointCloud()
        source.points = open3d.utility.Vector3dVector(xyz_worldCoor)
        source.colors = open3d.utility.Vector3dVector(rgb_colors)
        open3d.io.write_point_cloud("pcd/allPC.pcd", source)


    # INCREMENTAL PC
    elif mode == 1:

        xyz_worldCoor_temp = [[0, 0, 0]]
        rgb_colors_temp = [[0, 0, 0]]
        for i in range(1, 100, int(round(100/pc_viewnum))): 

            ### 이전에 생성된 pointcloud 점들을 해당 시점(i번째)로 투영하여 깊이영상과 색상영상 획득
            dist_mat, color_mat= projection_3dto2d(Rts[i], K, xyz_worldCoor, H, W, rgb_colors)

            ### PSNR 계산 (투영영상의 정확도 경향을 보기 위함)
            d = depth_imgs[i][...,:3]
            loss = np.mean(np.square(color_mat[np.logical_and(d<1000, d!=back_value)] - color_imgs[i][...,:3][np.logical_and(d<1000, d!=back_value)]))
            psnr = -10. * np.log10(loss)
            print("psnr:", psnr)

            ### visualization
            # plt.figure()
            # plt.imshow(depth_imgs[i])

            # plt.figure()
            # plt.imshow(dist_mat)

            # plt.figure()
            # plt.imshow(color_mat)

            # plt.figure()
            # plt.imshow(color_imgs[i])
            # plt.show()

            ### i번째 시점의 포인트클라우드 생성
            xyz_worldCoor_i, rgb_colors_i = reconstruction_2dto3d(color_imgs[i], depth_imgs[i], f, Rts[i], H, W) # 800x800x3

            ### PC를 투영하였을 때 투영된 점(depth thresholding 작업필요)과 투영되지 않은 점(recon작업 필요)
            ### i_t 인덱스가 투영된 점
            ### i_f 인덱스가 투영되지 않은 점
            i_t, j_t = np.where((dist_mat != 0) & (depth_imgs[i][...,0]<1000) & (depth_imgs[i][...,0] != back_value))
            i_f, j_f = np.where((dist_mat == 0) & (depth_imgs[i][...,0]<1000) & (depth_imgs[i][...,0] != back_value))

            ### projective matrix 계산
            Rt_inv = np.linalg.inv(Rts[i])
            P = K @ Rt_inv[:3, :]  #3x4
            
            ### homogeneous 좌표계로 변환 
            xyz1_t_worldCoor = np.concatenate([xyz_worldCoor_i[i_t, j_t], np.ones([xyz_worldCoor_i[i_t, j_t].shape[0], 1])], axis=-1) # N x 4
            
            rgb_t_colors = rgb_colors_i[i_t, j_t]

            ### 프로젝션된 점들과 카메라까지의 거리 계산
            w_mat = P[2,:] @ xyz1_t_worldCoor[..., np.newaxis] ## broacasting (1x4) dot (Nx4x1) => (Nx1) numpy gooood ~
            dist_origin = cal_dist_fromPointtoCam(w_mat, P)
            dist_origin = np.abs(dist_origin)
            
            ### Depth thresholding 
            ### Depth threshold보다 거리가 멀 경우, 새로운 점으로 저장
            k = np.where(abs(dist_mat[i_t, j_t] - dist_origin) > depth_threshold)[0]
            xyz_worldCoor = np.concatenate([xyz_worldCoor, xyz1_t_worldCoor[k][..., :3]], axis=0)
            rgb_colors = np.concatenate([rgb_colors, rgb_t_colors[k]], axis=0)
            xyz_worldCoor_temp = np.concatenate([xyz_worldCoor_temp, xyz1_t_worldCoor[k][...,:3]], axis=0)
            rgb_colors_temp = np.concatenate([rgb_colors_temp, rgb_t_colors[k]], axis=0)
            
            ### 투영이 되지 않는 점 (i_f) 새로운 점으로 저장
            xyz_worldCoor = np.concatenate([xyz_worldCoor, xyz_worldCoor_i[i_f, j_f]], axis=0)  #[Nx3]
            rgb_colors = np.concatenate([rgb_colors, rgb_colors_i[i_f, j_f]], axis=0)
            xyz_worldCoor_temp = np.concatenate([xyz_worldCoor_temp, xyz_worldCoor_i[i_f, j_f]], axis=0)
            rgb_colors_temp = np.concatenate([rgb_colors_temp, rgb_colors_i[i_f, j_f]], axis=0)

            print("xyz_worldCoor", xyz_worldCoor.shape)
            print("i :", i)

        ### 생성된 pointcloud를 pcd 파일로 저장
        source = open3d.geometry.PointCloud()
        source.points = open3d.utility.Vector3dVector(xyz_worldCoor)
        source.colors = open3d.utility.Vector3dVector(rgb_colors)
        open3d.io.write_point_cloud("pcd/incre_test.pcd", source)

    
    return xyz_worldCoor

def reconstruction_2dto3d(color_img, depth_img, f, Rt, H, W):

    x, y = np.meshgrid(tf.range(W, dtype=np.float32),
                    tf.range(H, dtype=np.float32), indexing='xy')  # i (W, H)
    
    ### inverse depth 일 경우
    # z = 1 - depth_img[...,0]
    ### inverse depth가 아닐 경우
    z =  depth_img[...,0]
    
    ### 3D reconstruction 수행
    xyz_camCoor = np.stack([(x-W*.5)/f * z, (y-H*.5)/f * -z, -z], -1) # (W, H, 3)
    xyz_camCoor = xyz_camCoor[..., np.newaxis]
    xyz1_camCoor = np.concatenate([xyz_camCoor, np.ones([800, 800, 1, 1])], axis=2) # N x 4
    xyz_worldCoor = (Rt[:3,:] @ xyz1_camCoor).squeeze()     # broadcasting (3x3) dot ((NxNx3x1) - (3x1)) => (NxNx3x1)   
    rgb_colors = color_img[..., :3]

    print("xyz_worldCoor", xyz_worldCoor.shape)

    return xyz_worldCoor, rgb_colors

def projection_3dto2d(Rt, K, xyz_worldCoor, H, W, rgb_colors=None):
    
    ### 투영 수행하여 2차원 픽셀 좌표 획득
    Rt_inv = np.linalg.inv(Rt)
    P = K @ Rt_inv[:3, :]
    xyz1_worldCoor = np.concatenate([xyz_worldCoor, np.ones([xyz_worldCoor.shape[0], 1])], axis=-1) # N x 4
    uvw_projected = (P @ xyz1_worldCoor[..., np.newaxis]).squeeze()     #broadcasting (3x4) dot (Nx4x1) => (N, 3, 1)

    uvw_projected[..., 0] = np.round(uvw_projected[..., 0] / uvw_projected[..., 2])
    uvw_projected[..., 1] = np.round(uvw_projected[..., 1] / uvw_projected[..., 2])

    #좌우반전
    uvw_projected[..., 0] = ((uvw_projected[..., 0] - W*.5) * -1 + W*.5)

    #####################

    # #깊이값 음값->양값으로 변환 (수정)
    # P_ = K @ Rt[:3, :] 
    # temp = np.stack([xyz1_worldCoor[..., 0],-xyz1_worldCoor[..., 1],-xyz1_worldCoor[..., 2], xyz1_worldCoor[..., 3]], axis=-1)
    # w_mat = P_[2,:] @ temp[...,None] ## broacasting (1x4) dot (Nx4x1) => (Nx1) numpy gooood ~
    # dist = cal_dist_fromPointtoCam(w_mat, P_)

    ##########################

    ### 거리값 획득 즉, 투영 깊이 영상의 픽셀값
    dist = cal_dist_fromPointtoCam(uvw_projected[..., 2], P)
    dist = np.abs(dist)
    
    uvw_projected = uvw_projected.astype(int)
    uvw_projected = np.reshape(uvw_projected, [-1,3])

    dist = np.reshape(dist, [-1])
    
    dist_mat = np.zeros((800, 800), dtype = np.float32)
    color_mat = np.zeros((800, 800, 3), dtype = np.float32)

    ### 투영 픽셀좌표 예외처리
    i_projected, = np.where((uvw_projected[:, 0]<H) &(uvw_projected[:, 1]<W)&(uvw_projected[:, 0]>0)&(uvw_projected[:, 1]>0))

    ### 이 과정은 최종 투영 영상을 획득
    ### 각 픽셀은 투영될 3차원 점 중 카메라로부터 가장 가까운 점이 투영되어야 함 
    ### 가장 가까운 3차원 점을 투영하여 최종 픽셀값 결정
    if rgb_colors is not None:
        for i in range(len(i_projected)):
            
            ### 현재 투영이 이루어지지 않았을 때 
            if dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] == 0 :
                dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] = dist[i_projected[i]]
                color_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] = rgb_colors[i_projected[i]]

            ### 이전에 투영된 점의 거리보다 i번째 점의 거리가 가까울 때 i번째 점으로 투영 결정
            elif dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] > dist[i_projected[i]]:
                dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] = dist[i_projected[i]]
                color_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] = rgb_colors[i_projected[i]]
        
        dist_mat = dist_mat.squeeze()
        return dist_mat, color_mat
        
    else : 
        for i in range(len(i_projected)):

            ### 현재 투영이 이루어지지 않았을 때 
            if dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] == 0 :
                dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] = dist[i_projected[i]]
            
            ### 이전에 투영된 점의 거리보다 i번째 점의 거리가 가까울 때 i번째 점으로 투영 결정
            elif dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] > dist[i_projected[i]]:
                dist_mat[uvw_projected[i_projected[i]][...,1], uvw_projected[i_projected[i]][...,0]] = dist[i_projected[i]]
        
        dist_mat = dist_mat.squeeze()
        return dist_mat, None

def make_filled_depthmap(Rt, focal, xyz_worldCoor, H, W, kernel_size, rgb_colors=None):
    print("kernel_size", kernel_size)
    K = np.array([[focal, 0, W/2.], [0, focal, H/2.], [0, 0, 1]])
    Rt = np.concatenate([Rt, [[0, 0, 0, 1]]], axis=0)

    ### 투영 깊이 영상 획득
    dist_mat, color_mat = projection_3dto2d(Rt, K, xyz_worldCoor, H, W,rgb_colors)
    
    ### 투영된 깊이 영상 holefilling 수행
    import copy
    dist_mat_nofilled = copy.deepcopy(dist_mat)
    dist_mat = hole_filling(dist_mat, kernel_size)

    # plt.figure()
    # plt.imshow(dist_mat_)
    # plt.figure()
    # plt.imshow(dist_mat)
    # plt.figure()
    # plt.imshow(color_mat)
    # plt.show()
    return dist_mat, color_mat, dist_mat_nofilled

def cal_dist_fromPointtoCam(w, P):
    # w Nx1?
    # P 3x4

    ### MVG 책 보세요
    M = P[:, :3]
    denom = np.sqrt(M[2, 0]*M[2, 0] + M[2, 1]*M[2, 1] + M[2, 2]*M[2, 2])

    if np.linalg.det(M) < 0 : 
        numer = -1
    elif np.linalg.det(M) == 0 : 
        numer = 0
    else :
        numer = 1

    numer = numer * w
    dist = numer / denom
    dist=dist.squeeze()
    return dist

def hole_filling(dist_mat, kernel_size):
    import copy
    
    dist_mat_filled = copy.deepcopy(dist_mat)

    # i_arr, j_arr, = np.where(np.logical_and(dist_mat == 0, depth_gt[..., 0]<1000))
    i_arr, j_arr, = np.where(dist_mat == 0)

    k = kernel_size #2,3,5
    H = dist_mat.shape[0]
    W = dist_mat.shape[1]
    
    ### 영상 가장자리 벗어나지 않도록 예외 처리
    for n in range(i_arr.shape[0]):
        if i_arr[n] > k-1: 
            i_arr= i_arr[n:]
            j_arr= j_arr[n:]
            break

    for n in range(j_arr.shape[0]):
        if j_arr[n] > k-1: 
            i_arr= i_arr[n:]
            j_arr= j_arr[n:]
            break

    for n in range(i_arr.shape[0], 0, -1): 
        if i_arr[n-1] < H - (k-1): 
            i_arr= i_arr[:n+1]
            j_arr= j_arr[:n+1]
            break

    for n in range(j_arr.shape[0], 0, -1): 
        if j_arr[n-1] < W - (k-1): 
            i_arr= i_arr[:n+1]
            j_arr= j_arr[:n+1]
            break

    i_arr_n, j_arr_n, = np.where(dist_mat != 0)
    i_arr_back = np.array([], dtype=np.int)
    j_arr_back = np.array([], dtype=np.int)

    ### 전경 부분에서 배경의 점을 추리는 과정
    for n in range(i_arr_n.shape[0]):
        ### 해당 픽셀 주변의 0이 아닌 픽셀들
        nonzero_values = dist_mat[i_arr_n[n]-k:i_arr_n[n]+k+1, j_arr_n[n]-k:j_arr_n[n]+k+1][dist_mat[i_arr_n[n]-k:i_arr_n[n]+k+1, j_arr_n[n]-k:j_arr_n[n]+k+1] != 0.]
        mean = np.mean(nonzero_values)
        std = np.std(nonzero_values)
        if abs(dist_mat[i_arr_n[n], j_arr_n[n]]-mean) > 2* std : 
            i_arr_back = np.concatenate([i_arr_back, [i_arr_n[n]]])
            j_arr_back = np.concatenate([j_arr_back, [j_arr_n[n]]])
    
    ### 홀 주변값 중에서 홀이 아닌 값들의 평균으로 값 채우기 
    for n in range(i_arr.shape[0]):
        nonzero_values = dist_mat[i_arr[n]-k:i_arr[n]+k+1, j_arr[n]-k:j_arr[n]+k+1][dist_mat[i_arr[n]-k:i_arr[n]+k+1, j_arr[n]-k:j_arr[n]+k+1] != 0.]
        dist_mat_filled[i_arr[n], j_arr[n]] = np.where(nonzero_values.shape[0]>0, np.mean(nonzero_values), 0.)

    for n in range(i_arr_back.shape[0]):
        nonzero_values = dist_mat[i_arr_back[n]-k:i_arr_back[n]+k+1, j_arr_back[n]-k:j_arr_back[n]+k+1][dist_mat[i_arr_back[n]-k:i_arr_back[n]+k+1, j_arr_back[n]-k:j_arr_back[n]+k+1] != 0.]
        dist_mat_filled[i_arr_back[n], j_arr_back[n]] = np.where(nonzero_values.shape[0]>0, np.mean(nonzero_values), 0.)
    
    return dist_mat_filled
