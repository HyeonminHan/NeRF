from scipy.stats import skewnorm
import matplotlib.pyplot as plt
import sys
import numpy as np
import imageio
import OpenEXR
import Imath
from PIL import Image
import statistics

# '''
#exr 파일 jpg로 저장 
# for i in range(10):
#     fname_d = 'temp/r_'+str(i)+'_depth_0001.exr'
#     exr = imageio.imread(fname_d, format='EXR-FI')
#     exr = np.array(exr)
#     jpg = (255*np.clip(exr / 8.0, 0, 1)).astype(np.uint8)
#     imageio.imwrite('temp/r_'+str(i)+'_depth_0001.jpg', jpg)
# '''

# '''
fname_d = 'temp/051.png'
exr = imageio.imread(fname_d)
jpg = np.array(exr)
# jpg = (255*np.clip(exr / 8.0, 0, 1)).astype(np.uint8)

#배경 검정으로 바꾸기
jpg[np.logical_and(jpg[...,0]==255, jpg[...,1]==255, jpg[...,2]==255)] =0
imageio.imwrite('temp/051_.jpg', jpg)
# '''

'''
#배경을 제외한 PSNR 구하기
psnrs = []
for i in range(60):
    fname_estimated = 'temp/estimated/_1_'+ str(i)+'.png'
    if i <10:
        fname_gt = 'temp/gt/0000'+ str(i)+'.png'
    else :
        fname_gt = 'temp/gt/000'+ str(i)+'.png'

    img_gt = imageio.imread(fname_gt)
    img_estimated = imageio.imread(fname_estimated)

    img_gt = (np.array(img_gt) / 255.).astype(np.float32)
    img_estimated = (np.array(img_estimated) / 255.).astype(np.float32)
    loss = np.mean(np.square(img_estimated[np.logical_and(img_gt>0., img_gt<1.)] - img_gt[np.logical_and(img_gt>0., img_gt<1.)]))
    psnr = -10. * np.log10(loss)
    print("img_gt:", img_gt[np.logical_and(img_gt>0., img_gt<1.)])

    print("psnr " + str(i)+":", psnr)
    psnrs.append(psnr)

print("mean psnr:", statistics.mean(psnrs))
'''

'''
#배경 날리기
fname_d = "temp/depth_est_037.png"
jpg = imageio.imread(fname_d)
jpg = np.array(jpg)
jpg = np.where(jpg==0., 255, jpg)
imageio.imwrite("temp/depth_est_037_nobackgd.png", jpg)
'''
'''
#홀필링
fname_d = "temp/depth_est_nofill_002_pc50.png"
jpg = imageio.imread(fname_d)

def hole_filling(dist_mat, kernel_size):
    import copy
    
    dist_mat_filled = copy.deepcopy(dist_mat)

    # i_arr, j_arr, = np.where(np.logical_and(dist_mat == 0, depth_gt[..., 0]<1000))
    i_arr, j_arr, = np.where(dist_mat == 0)

    k = kernel_size #2,3,5
    H = dist_mat.shape[0]
    W = dist_mat.shape[1]
    
    # 영상 가장자리 벗어나지 않도록 예외 처리
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

    #전경 부분에서 배경의 점을 추리는 과정
    for n in range(i_arr_n.shape[0]):
        #해당 픽셀 주변의 0이 아닌 픽셀들
        nonzero_values = dist_mat[i_arr_n[n]-k:i_arr_n[n]+k+1, j_arr_n[n]-k:j_arr_n[n]+k+1][dist_mat[i_arr_n[n]-k:i_arr_n[n]+k+1, j_arr_n[n]-k:j_arr_n[n]+k+1] != 0.]
        mean = np.mean(nonzero_values)
        std = np.std(nonzero_values)
        if abs(dist_mat[i_arr_n[n], j_arr_n[n]]-mean) > 2* std : 
            i_arr_back = np.concatenate([i_arr_back, [i_arr_n[n]]])
            j_arr_back = np.concatenate([j_arr_back, [j_arr_n[n]]])
    
    #홀 주변값 중에서 홀이 아닌 값들의 평균으로 값 채우기 
    for n in range(i_arr.shape[0]):
        nonzero_values = dist_mat[i_arr[n]-k:i_arr[n]+k+1, j_arr[n]-k:j_arr[n]+k+1][dist_mat[i_arr[n]-k:i_arr[n]+k+1, j_arr[n]-k:j_arr[n]+k+1] != 0.]
        dist_mat_filled[i_arr[n], j_arr[n]] = np.where(nonzero_values.shape[0]>0, np.mean(nonzero_values), 0.)

    for n in range(i_arr_back.shape[0]):
        nonzero_values = dist_mat[i_arr_back[n]-k:i_arr_back[n]+k+1, j_arr_back[n]-k:j_arr_back[n]+k+1][dist_mat[i_arr_back[n]-k:i_arr_back[n]+k+1, j_arr_back[n]-k:j_arr_back[n]+k+1] != 0.]
        dist_mat_filled[i_arr_back[n], j_arr_back[n]] = np.where(nonzero_values.shape[0]>0, np.mean(nonzero_values), 0.)
    
    return dist_mat_filled

jpg = hole_filling(jpg, 5)

jpg = np.array(jpg)
imageio.imwrite("temp/depth_est_nofill_002_filled_pc50.png", jpg)
'''
exit()

fig, ax = plt.subplots(1, 1)


    
# numargs = skewnorm .numargs 
# a, b = 4.32, 3.18
# rv = skewnorm (a, b) 
    
# print ("RV : \n", rv)

a = -2
mean, var, skew, kurt = skewnorm.stats(a, moments='mvsk')


# print("mean:", mean,"var:", var, "skew:", skew, "kurt:", kurt)
# print("skewnorm.ppf(0.01, a):", skewnorm.ppf(0.01, a))
# print("skewnorm.ppf(0.99, a):", skewnorm.ppf(0.99, a))

# print("a:", a)
# x = np.linspace(skewnorm.ppf(0.01, a),
#                 skewnorm.ppf(0.99, a), 100)
# print("x;", x)
# print("skewnorm.pdf(x, a):", skewnorm.pdf(x, a))

# ax.plot(x, skewnorm.pdf(x, a),
#        'r-', lw=5, alpha=0.6, label='skewnorm pdf')
# rv = skewnorm(a)
# ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
# vals = skewnorm.ppf([0.001, 0.5, 0.999], a)
# print("dd:", np.allclose([0.001, 0.5, 0.999], skewnorm.cdf(vals, a)))
# print("a2:", a)
r = skewnorm.rvs(a, size=64)
r = 0.15 * r + 3.1
print("r :", r)
# ax.hist(r, normed=True, histtype='stepfilled', alpha=0.2)
ax.hist(r, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()