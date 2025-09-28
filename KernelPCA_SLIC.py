import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift
from sklearn import preprocessing
import cv
import math

seed_gpu = 0
np.random.seed(seed_gpu)


def LSC_superpixel(I, nseg):
    superpixelNum = nseg
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv.cvtColor(I, cv.COLOR_BGR2HSV)
    height, width, channels = I_new.shape

    superpixelNum = nseg
    seeds = cv.ximgproc.createSuperpixelSEEDS(width, height, channels,
                                              int(superpixelNum),
                                              num_levels=2, prior=1,
                                              histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
    return segments


def SegmentsLabelProcess(labels):
    """
    对labels做后处理，防止出现label不连续现象
    """
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels


class SLIC(object):
    """
    与原先 LDA_SLIC 中的 SLIC 类一致：
    - 做标准化
    - 调用 skimage.segmentation.slic
    - 计算 Q, S, segments, 以及邻接矩阵 A
    """
    def __init__(self, HSI, labels, n_segments=1000, compactness=20,
                 max_iter=20, sigma=0, min_size_factor=0.3, max_size_factor=2):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma

        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands])
        self.labels = labels

    def get_Q_and_S_and_Segments(self):
        img = self.data
        (h, w, d) = img.shape
        segments = slic(img,
                        n_segments=self.n_segments,
                        compactness=self.compactness,
                        max_iter=self.max_iter,
                        convert2lab=False,
                        sigma=self.sigma,
                        enforce_connectivity=True,
                        min_size_factor=self.min_size_factor,
                        max_size_factor=self.max_size_factor,
                        slic_zero=False)

        # 若 label 不连续，修正
        if segments.max() + 1 != len(set(np.reshape(segments, [-1]).tolist())):
            segments = SegmentsLabelProcess(segments)

        self.segments = segments
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        print("superpixel_count =", superpixel_count)

        # segments 展平
        segments = np.reshape(segments, [-1])
        S = np.zeros([superpixel_count, d], dtype=np.float32)
        Q = np.zeros([h * w, superpixel_count], dtype=np.float32)

        x = np.reshape(img, [-1, d])
        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel_mean = np.sum(pixels, axis=0) / count
            S[i] = superpixel_mean
            Q[idx, i] = 1

        self.S = S
        self.Q = Q
        return Q, S, np.reshape(self.segments, [h, w])

    def get_A(self, sigma: float):
        A = np.zeros([self.superpixel_count, self.superpixel_count],
                     dtype=np.float32)
        (h, w) = self.segments.shape
        for i in range(h - 1):
            for j in range(w - 1):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:
                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue
                    pix1 = self.S[idx1]
                    pix2 = self.S[idx2]
                    dist = np.sum((pix1 - pix2)**2)
                    diss = np.exp(-dist / (sigma**2))
                    A[idx1, idx2] = A[idx2, idx1] = diss
        return A


class KernelPCA_SLIC(object):
    """
    用 Kernel PCA 替换原先 LDA 的逻辑。
    - n_component: 保留的主成分数
    - kernel: 'rbf'/'poly'/'sigmoid'/... 可改
    - gamma 等核参数需自行调节
    """
    def __init__(self, data, labels, n_component):
        """
        :param data: 原始高光谱数据, shape=[height, width, bands]
        :param labels: 训练标签, shape=[height, width]
        :param n_component: Kernel PCA 保留的维度
        """
        self.data = data
        self.init_labels = labels
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape

        # 将原图打平
        self.x_flatt = np.reshape(data, [self.height * self.width, self.bands])
        self.labels = labels

    def KernelPCA_Process(self, curr_labels):
        """
        使用 Kernel PCA 进行降维。
        如果想和 LDA 保持“只用有标注像素来 fit”的做法，可如下：
          1) 取出 labels != 0 的像素进行 fit
          2) 再对整幅图 transform
        如果想用全图 fit，就不需要 idx.
        """
        curr_labels = np.reshape(curr_labels, [-1])
        idx = np.where(curr_labels != 0)[0]
        if len(idx) == 0:
            # 如果没有标注像素，就直接对全图做 Kernel PCA
            x_for_fit = self.x_flatt
        else:
            x_for_fit = self.x_flatt[idx]

        # 你可以修改 kernel='rbf' 为其他，如 'poly'、'sigmoid' 等
        # gamma 默认为 1/n_features，这里可自行调参
        kPCA = KernelPCA(n_components=self.n_component,
                         kernel='rbf',
                         gamma=None,        # 可改成 1/(2*sigma^2) 等
                         fit_inverse_transform=False,
                         eigen_solver='auto')

        kPCA.fit(x_for_fit)
        # 再 transform 整幅图
        X_new = kPCA.transform(self.x_flatt)
        # 变回 (height, width, n_component)
        return np.reshape(X_new, [self.height, self.width, -1])

    def SLIC_Process(self, img, scale=25):
        """
        做 SLIC 并得到 Q, S, A, segments
        """
        n_segments_init = int((self.height * self.width) / scale)
        print("n_segments_init =", n_segments_init)

        slic_model = SLIC(
            HSI=img,
            labels=self.labels,
            n_segments=n_segments_init,
            compactness=1,
            sigma=1,
            min_size_factor=0.1,
            max_size_factor=2
        )
        Q, S, segments = slic_model.get_Q_and_S_and_Segments()
        A = slic_model.get_A(sigma=10)
        return Q, S, A, segments

    def simple_superpixel(self, scale):
        """
        先对原始数据做 Kernel PCA，再做 SLIC 分割
        """
        curr_labels = self.init_labels
        # 原先用 LDA_Process，这里改为 KernelPCA_Process
        X = self.KernelPCA_Process(curr_labels)
        Q, S, A, Seg = self.SLIC_Process(X, scale=scale)
        return Q, S, A, Seg, X

    def simple_superpixel_no_KernelPCA(self, scale):
        """
        直接对原始数据做 SLIC，不降维
        """
        Q, S, A, Seg = self.SLIC_Process(self.data, scale=scale)
        return Q, S, A, Seg, X

