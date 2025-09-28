import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
import time
from sklearn import preprocessing
import torch
import KernelPCA_SLIC
import GTCFN
import torch.backends.cudnn as cudnn
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
import matplotlib
from matplotlib.colors import ListedColormap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择cpu或者GPU
seed_gpu = 0
random.seed(seed_gpu)
np.random.seed(seed_gpu)
torch.manual_seed(seed_gpu)
torch.cuda.manual_seed(seed_gpu)
cudnn.deterministic = True
cudnn.benchmark = False


def count_parameters(model):
    """
    统计模型中“可训练参数（requires_grad=True）”的数量
    返回值是一个整数，如 123456，代表这个模型总共有 123,456 个可训练参数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

####数据集选择#####
# FLAG =1, indian
# FLAG =2, paviaU
# FLAG =3, salinas
samples_type = 'same_num'  # samples_type类型为ratio

for (FLAG, Scale) in [(1, 600)]:
    # for (FLAG, curr_train_ratio,Scale) in [(2,0.01,100),(3,0.01,100)]:
    torch.cuda.empty_cache()  # pytorch的显存释放机制torch.cuda.empty_cache () Pytorch已经可以自动回收我们不用的显存
    #######一些质量指标#######
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []


    Seed_List = [1330, 1330, 1332]  # SA
    DATASET = 'indian'
    ####读取训练集设置相关参数####
    if DATASET == 'paviaU':
        data_mat = sio.loadmat('./HyperImage_data/paviaU/PaviaU.mat')
        data = data_mat['paviaU']
        gt_mat = sio.loadmat('./HyperImage_data/paviaU/PaviaU_gt.mat')
        gt = gt_mat['paviaU_gt']
        dataset_name = "paviaU_"  # 数据集名称
        class_count = 9  # 样本类别数
        val_ratio = 0.002  # 验证集比例
    elif DATASET == 'indian':
        data_mat = sio.loadmat(
            './HyperImage_data/indian/Indian_pines_corrected.mat')
        data = data_mat['indian_pines_corrected']
        gt_mat = sio.loadmat('./HyperImage_data/indian/Indian_pines_gt.mat')
        gt = gt_mat['indian_pines_gt']
        dataset_name = "indian_"  # 数据集名称
        class_count = 16  # 样本类别数
        val_ratio = 0.02  # 验证集比例
    elif DATASET == 'salinas':
        data_mat = sio.loadmat('./HyperImage_data/Salinas/Salinas_corrected.mat')
        data = data_mat['salinas_corrected']
        gt_mat = sio.loadmat('./HyperImage_data/Salinas/Salinas_gt.mat')
        gt = gt_mat['salinas_gt']
        dataset_name = "salinas_"  # 数据集名称
        class_count = 16  # 样本类别数
        val_ratio = 0.002  # 验证集比例

    if FLAG == 1:
        learning_rate = 3e-4  # 学习率
        max_epoch = 300  # 迭代次数

        pass

    superpixel_scale = Scale  #########################
    train_samples_per_class = 30 # 训练样本所占的比例
    val_samples = class_count  # 验证集样本的种类数
    cmap = cm.get_cmap('jet', class_count + 1)  # 拿到想要的 colormap，然后给其传入数值就会返回 rgb
    plt.set_cmap(cmap)  # 设置绘图颜色
    m, n, d = data.shape  # 高光谱数据的三个维度 145*145*200

    ############ 数据standardization标准化,即提前全局BN#########
    orig_data = data  # 原始数据集11
    height, width, bands = data.shape  # 原始高光谱数据的三个维度
    data = np.reshape(data, [height * width, bands])  # 数据由145*145*200到21025*200
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)  # 计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
    data = np.reshape(data, [height, width, bands])


    ########经过上述操作，数据格式不变145*145*200，但是数值变为标准正态分布##########
    my_colors = ['white','Coral', 'darkorange',  'gold',  'limegreen',  'green',  'LawnGreen',  'Turquoise', 'cyan', 'PowDerBlue', 'DeepSkyBlue', 'LightSkyBlue', 'mediumorchid','plum', 'HotPink', 'magenta',    'pink'
]
    my_cmap = ListedColormap(my_colors)

    def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
        '''
        get classification map , then save to given path
        :param label: classification label, 2D
        :param name: saving path and file's name
        :param scale: scale of image. If equals to 1, then saving-size is just the label-size
        :param dpi: default is OK
        :return: null
        '''
        fig, ax = plt.subplots()
        numlabel = np.array(label, dtype=int)
        plt.imshow(numlabel, cmap=my_cmap, interpolation='none',
                   vmin=0, vmax=class_count)
        ax.set_axis_off()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
        foo_fig = plt.gcf()  # 'get current figure'
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
        pass


    def GT_To_One_Hot(gt, class_count):
        '''
        Convet Gt to one-hot labels
        :param gt:
        :param class_count:
        :return:
        '''
        GT_One_Hot = []  # 转化为one-hot形式的标签
        for i in range(gt.shape[0]):
            for j in range(gt.shape[1]):
                temp = np.zeros(class_count, dtype=np.float32)
                if gt[i, j] != 0:
                    temp[int(gt[i, j]) - 1] = 1
                GT_One_Hot.append(temp)
        GT_One_Hot = np.reshape(GT_One_Hot, [height, width, class_count])
        return GT_One_Hot


    gt_reshape = np.reshape(gt, [-1])  # 标签从145*145*1变为21015*1
    train_rand_idx = np.empty(shape=(0, 0))
    mm = []

    train_rand_idx = []
    # train_data_index = []
    # val_rand_idx = []
    # rand_idx = []
    val_rand_idx = []

    if samples_type == 'same_num':
        for i in range(class_count):
            idx = np.where(gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            real_train_samples_per_class = train_samples_per_class
            rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
            # if real_train_samples_per_class > samplesCount:
            #     real_train_samples_per_class = samplesCount
            if samplesCount <= 50:
                real_train_samples_per_class = 15
            real_train_samples_per_class = min(real_train_samples_per_class, samplesCount)
            rand_idx = random.sample(rand_list,
                                     real_train_samples_per_class)  # 随机数数量 四舍五入(改为上取整)
            rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
            train_rand_idx.append(rand_real_idx_per_class_train)
        train_rand_idx = np.array(train_rand_idx)
        val_rand_idx = np.array(val_rand_idx)
        train_data_index = []
        for c in range(train_rand_idx.shape[0]):
            a = train_rand_idx[c]
            for j in range(a.shape[0]):
                train_data_index.append(a[j])
        train_data_index = np.array(train_data_index)

        train_data_index = set(train_data_index)
        all_data_index = [i for i in range(len(gt_reshape))]
        all_data_index = set(all_data_index)

        # 背景像元的标签
        background_idx = np.where(gt_reshape == 0)[-1]
        background_idx = set(background_idx)
        test_data_index = all_data_index - train_data_index - background_idx


        val_data_count = 30  # 需求：验证集中总共要抽30个
        test_data_index = list(test_data_index)  # 如果它不是列表，就转为列表

        # 1. 先把“测试集中”每个类别的像素索引都分门别类存起来
        test_idx_per_class = [[] for _ in range(class_count)]
        for idx in test_data_index:
            label_val = gt_reshape[idx]
            if label_val != 0:  # 如果不是背景
                class_id = int(label_val) - 1
                test_idx_per_class[class_id].append(idx)

        # 2. 每个类别至少抽 1 个做验证
        val_data_index = []
        for c in range(class_count):
            if len(test_idx_per_class[c]) > 0:
                pick_idx = random.choice(test_idx_per_class[c])  # 从当前类别的测试索引中随机取1个
                val_data_index.append(pick_idx)
                # 从 test_idx_per_class[c] 移除这个已经抽到验证集的像素，防止后面重复抽
                test_idx_per_class[c].remove(pick_idx)

        # 3. 上一步总共已经抽了 "class_count" 个 (对 Indian Pines 就是16个)
        #    那么剩余还需要抽 (30 - class_count) 个，一次性在剩下的所有测试像素里随机抽就行
        already_picked = len(val_data_index)  # 已经抽了多少验证样本
        left_to_pick = val_data_count - already_picked
        if left_to_pick < 0:
            left_to_pick = 0  # 理论上不会小于0，但防止万一

        # 汇总剩下的所有测试像素(不分类别)
        remaining_test_pool = []
        for c in range(class_count):
            remaining_test_pool.extend(test_idx_per_class[c])

        if left_to_pick <= len(remaining_test_pool):
            val_data_index.extend(random.sample(remaining_test_pool, left_to_pick))
        else:
            # 如果测试集中可抽样本还不够，那么就只能全部拿来做验证(相当于验证集 < 50)
            val_data_index.extend(remaining_test_pool)

        # 4. 最终把 val_data_index 转为 set，并更新 test_data_index
        val_data_index = set(val_data_index)
        test_data_index = set(test_data_index) - val_data_index

        # 将训练集 验证集 测试集 整理
        test_data_index = list(test_data_index)
        train_data_index = list(train_data_index)
        val_data_index = list(val_data_index)
    # 获取训练样本的标签图
    train_samples_gt = np.zeros(gt_reshape.shape)  # 得到21015*1的零数组
    for i in range(len(train_data_index)):
        train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
        pass
    #######经过上述操作后train_samples_gt在训练集处获得标签，其余为0########

    # 获取测试样本的标签图
    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_data_index)):
        test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
        pass
    #######经过上述操作后test_samples_gt在训练集处获得标签，其余为0########

    # 下面的代码放在您分好集(train_data_index, test_data_index 等)之后：
    train_count_per_class = [0] * class_count
    test_count_per_class = [0] * class_count
    val_count_per_class = [0] * class_count

    # 先把它们转换成 set，方便“是否包含”判断
    train_data_index_set = set(train_data_index)
    test_data_index_set = set(test_data_index)
    val_data_index_set = set(val_data_index)

    # 遍历所有像元
    for px_idx in range(len(gt_reshape)):
        label = gt_reshape[px_idx]
        if label != 0:  # 背景为 0，不算
            class_id = int(label) - 1  # 类别从1开始，这里转为0开头
            if px_idx in train_data_index_set:
                train_count_per_class[class_id] += 1
            elif px_idx in test_data_index_set:
                test_count_per_class[class_id] += 1
            elif px_idx in val_data_index_set:
                val_count_per_class[class_id] += 1

    # 打印结果
    for c in range(class_count):
        print(f"类别 {c + 1} : 训练集数量 = {train_count_per_class[c]}, "
              f"验证集数量 = {val_count_per_class[c]}, "
              f"测试集数量 = {test_count_per_class[c]}")

    Test_GT = np.reshape(test_samples_gt, [m, n])  # 将标签图从21025*1转化为145*145*1

    # 获取验证集样本的标签图
    val_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(val_data_index)):
        val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
        pass
    #######经过上述操作后val_samples_gt在测试集处获得标签，其余为0########

    train_samples_gt = np.reshape(train_samples_gt, [height, width])
    test_samples_gt = np.reshape(test_samples_gt, [height, width])
    val_samples_gt = np.reshape(val_samples_gt, [height, width])
    #######经过上述操作后标签图从21025*1转化为145*145*1########

    train_samples_gt_onehot = GT_To_One_Hot(train_samples_gt, class_count)
    test_samples_gt_onehot = GT_To_One_Hot(test_samples_gt, class_count)
    val_samples_gt_onehot = GT_To_One_Hot(val_samples_gt, class_count)
    #######经过上述操作后标签图从145*145*1转化为145*145*16########
    train_samples_gt_onehot = np.reshape(train_samples_gt_onehot, [-1, class_count]).astype(int)
    test_samples_gt_onehot = np.reshape(test_samples_gt_onehot, [-1, class_count]).astype(int)
    val_samples_gt_onehot = np.reshape(val_samples_gt_onehot, [-1, class_count]).astype(int)
    #######经过上述操作后所有标签图从145*145*1转化为21025*16 16是独热码，代表物体种类########

    ############制作训练数据和测试数据的gt掩膜.根据GT将带有标签的像元设置为全1向量##############
    # 训练集
    train_label_mask = np.zeros([m * n, class_count])
    temp_ones = np.ones([class_count])
    train_samples_gt = np.reshape(train_samples_gt, [m * n])
    for i in range(m * n):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [m * n, class_count])
    ############经过上述操作得到训练集gt掩膜.根据GT将带有标签的像元设置为全1向量，145*145*16的掩码，有物体的为全1其余为全0##############

    # 测试集
    test_label_mask = np.zeros([m * n, class_count])
    temp_ones = np.ones([class_count])
    test_samples_gt = np.reshape(test_samples_gt, [m * n])
    for i in range(m * n):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [m * n, class_count])
    ############经过上述操作得到测试集gt掩膜.根据GT将带有标签的像元设置为全1向量，145*145*16的掩码，有物体的为全1其余为全0##############

    # 验证集
    val_label_mask = np.zeros([m * n, class_count])
    temp_ones = np.ones([class_count])
    val_samples_gt = np.reshape(val_samples_gt, [m * n])
    for i in range(m * n):
        if val_samples_gt[i] != 0:
            val_label_mask[i] = temp_ones
    val_label_mask = np.reshape(val_label_mask, [m * n, class_count])
    ############经过上述操作得到验证集gt掩膜.根据GT将带有标签的像元设置为全1向量，145*145*16的掩码，有物体的为全1其余为全0##############

    ###LDA 线性判别分析    https://zhuanlan.zhihu.com/p/137968371
    ls = KernelPCA_SLIC.KernelPCA_SLIC(
        data,
        np.reshape(train_samples_gt, [height, width]),
        n_component=15  # 或者换成你需要的维度
    )
    tic0 = time.time()
    Q, S, A, Seg, X_new = ls.simple_superpixel(scale=superpixel_scale)
    toc0 = time.time()
    LDA_SLIC_Time = toc0 - tic0
    # np.save(dataset_name+'Seg',Seg)
    print("LDA-SLIC costs time: {}".format(LDA_SLIC_Time))
    Q = torch.from_numpy(Q).to(device)



    def adj_mul(adj_i, adj, N):
        adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
        adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
        adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
        adj_j = adj_j.coalesce().indices()
        return adj_j


    def get_adj(A):
        n_node, _ = A.shape  # 196
        n_edge = np.count_nonzero(A)  # 1046
        adj = np.zeros((2, n_edge), dtype=int)
        m = 0
        for i in range(n_node):
            for j in range(n_node):
                if A[i, j] != 0:
                    adj[0, m] = i
                    adj[1, m] = j
                    m = m + 1
        adj = torch.from_numpy(adj).to(device)
        adjs = []
        adj, _ = add_self_loops(adj, num_nodes=n_node)  # 边数=1046/2 双边数加节点数 1046+196=1242
        adjs.append(adj)
        for i in range(1 - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, n_node)
            adjs.append(adj)

        return adjs


    adj = get_adj(A)

    A = torch.from_numpy(A).to(device)

    # 转到GPU
    train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
    test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)
    val_samples_gt = torch.from_numpy(val_samples_gt.astype(np.float32)).to(device)
    # 转到GPU
    train_samples_gt_onehot = torch.from_numpy(train_samples_gt_onehot.astype(np.float32)).to(device)
    test_samples_gt_onehot = torch.from_numpy(test_samples_gt_onehot.astype(np.float32)).to(device)
    val_samples_gt_onehot = torch.from_numpy(val_samples_gt_onehot.astype(np.float32)).to(device)
    # 转到GPU
    train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
    test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)
    val_label_mask = torch.from_numpy(val_label_mask.astype(np.float32)).to(device)

    net_input = np.array(data, np.float32)
    net_input = torch.from_numpy(net_input.astype(np.float32)).to(device)

    for curr_seed in Seed_List:  # Seed_List=[0,1,2,3,4]#随机种子点


        if dataset_name == "indian_":
            net = GTCFN.GTCFN(height, width, bands, class_count, Q, A, adj, model='smoothed')
        else:
            net = GTCFN.GTCFN(height, width, bands, class_count, Q, A, adj)

        # 调用统计函数，打印可训练参数的数量
        print("Number of trainable parameters (可训练参数总数):", count_parameters(net))
        print("parameters", net.parameters(), len(list(net.parameters())))
        net.to(device)




        def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor, reallabel_mask: torch.Tensor):
            real_labels = reallabel_onehot  # 21025*16 独热码
            we = -torch.mul(real_labels, torch.log(predict))  # 21025*16 *21025*16 预测值和标签点乘
            we = torch.mul(we, reallabel_mask)  # 训练蒙板，只计算训练样本的损失函数
            pool_cross_entropy = torch.sum(we)  # 计算损失函数的值
            return pool_cross_entropy


        zeros = torch.zeros([m * n]).to(device).float()


        def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, require_AA_KPP=False,
                                 printFlag=True):
            if False == require_AA_KPP:
                with torch.no_grad():
                    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()  # 有效标签的个数
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count

                    return OA
            else:
                with torch.no_grad():
                    # 计算OA
                    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
                    available_label_count = available_label_idx.sum()  # 有效标签的个数
                    correct_prediction = torch.where(
                        torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1),
                        available_label_idx, zeros).sum()
                    OA = correct_prediction.cpu() / available_label_count
                    OA = OA.cpu().numpy()

                    # 计算AA
                    zero_vector = np.zeros([class_count])
                    output_data = network_output.cpu().numpy()
                    train_samples_gt = train_samples_gt.cpu().numpy()
                    train_samples_gt_onehot = train_samples_gt_onehot.cpu().numpy()

                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    for z in range(output_data.shape[0]):
                        if ~(zero_vector == output_data[z]).all():
                            idx[z] += 1
                    # idx = idx + train_samples_gt
                    count_perclass = np.zeros([class_count])
                    correct_perclass = np.zeros([class_count])
                    for x in range(len(train_samples_gt)):
                        if train_samples_gt[x] != 0:
                            count_perclass[int(train_samples_gt[x] - 1)] += 1
                            if train_samples_gt[x] == idx[x]:
                                correct_perclass[int(train_samples_gt[x] - 1)] += 1
                    test_AC_list = correct_perclass / count_perclass
                    test_AA = np.average(test_AC_list)

                    # 计算KPP
                    test_pre_label_list = []
                    test_real_label_list = []
                    output_data = np.reshape(output_data, [m * n, class_count])
                    idx = np.argmax(output_data, axis=-1)
                    idx = np.reshape(idx, [m, n])
                    for ii in range(m):
                        for jj in range(n):
                            if Test_GT[ii][jj] != 0:
                                test_pre_label_list.append(idx[ii][jj] + 1)
                                test_real_label_list.append(Test_GT[ii][jj])
                    test_pre_label_list = np.array(test_pre_label_list)
                    test_real_label_list = np.array(test_real_label_list)
                    kappa = metrics.cohen_kappa_score(test_pre_label_list.astype(np.int16),
                                                      test_real_label_list.astype(np.int16))
                    test_kpp = kappa

                    # 输出
                    if printFlag:
                        print("test OA=", OA, "AA=", test_AA, 'kpp=', test_kpp)
                        print('acc per class:')
                        print(test_AC_list)

                    OA_ALL.append(OA)
                    AA_ALL.append(test_AA)
                    KPP_ALL.append(test_kpp)
                    AVG_ALL.append(test_AC_list)

                    # 保存数据信息
                    f = open('results\\' + dataset_name + '_results.txt', 'a+')
                    str_results = '\n======================' \
                                  + " learning rate=" + str(learning_rate) \
                                  + " epochs=" + str(max_epoch) \
                                  + " val ratio=" + str(val_ratio) \
                                  + " ======================" \
                                  + "\nOA=" + str(OA) \
                                  + "\nAA=" + str(test_AA) \
                                  + '\nkpp=' + str(test_kpp) \
                                  + '\nacc per class:' + str(test_AC_list) + "\n"
                    # + '\ntrain time:' + str(time_train_end - time_train_start) \
                    # + '\ntest time:' + str(time_test_end - time_test_start) \
                    f.write(str_results)
                    f.close()
                    return OA


        # 训练
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # ,weight_decay=0.0001
        best_loss = 99999
        net.train()
        tic1 = time.clock()
        for i in range(max_epoch + 1):
            optimizer.zero_grad()  # zero the gradient buffers
            output = net(net_input)  # 145*145*200到21025*16
            loss = compute_loss(output, train_samples_gt_onehot,
                                train_label_mask)  # train_samples_gt_onehot 21025*16 是独热码  21025*16 全1或全0
            loss.backward(retain_graph=False)
            optimizer.step()  # Does the update
            if i % 50 == 0:
                with torch.no_grad():
                    net.eval()
                    output = net(net_input)
                    trainloss = compute_loss(output, train_samples_gt_onehot, train_label_mask)
                    trainOA = evaluate_performance(output, train_samples_gt, train_samples_gt_onehot)
                    valloss = compute_loss(output, val_samples_gt_onehot, val_label_mask)
                    valOA = evaluate_performance(output, val_samples_gt, val_samples_gt_onehot)
                    print(
                        "{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                         valloss, valOA))

                    if valloss < best_loss:
                        best_loss = valloss
                        torch.save(net.state_dict(), "model\\best_model.pt")
                        print('save model...')
                torch.cuda.empty_cache()
                net.train()
        toc1 = time.clock()
        print("\n\n====================training done. starting evaluation...========================\n")
        training_time = toc1 - tic1 + LDA_SLIC_Time  # 分割耗时需要算进去
        Train_Time_ALL.append(training_time)

        torch.cuda.empty_cache()
        with torch.no_grad():
            net.load_state_dict(torch.load("model\\best_model.pt"))
            net.eval()
            tic2 = time.clock()
            output = net(net_input)
            toc2 = time.clock()
            testloss = compute_loss(output, test_samples_gt_onehot, test_label_mask)
            testOA = evaluate_performance(output, test_samples_gt, test_samples_gt_onehot, require_AA_KPP=True,
                                          printFlag=False)
            print("{}\ttest loss={}\t test OA={}".format(str(i + 1), testloss, testOA))
            # 计算
            # —— 全图预测 + 掩掉背景 ——
            # 1) 扁平化预测结果
            pred_flat = torch.argmax(output, dim=1).cpu().numpy()  # 长度 m*n

            # 2) 准备一个全零背景的扁平化图
            classification_flat = np.zeros(m * n, dtype=np.uint8)

            # 3) 把所有原始标签中非背景的位置填上预测值 (+1 恢复原始标签编号)
            gt_flat = gt.reshape(-1)  # 假设 gt 为原始 ground-truth array，shape (m, n)
            mask_idx = np.where(gt_flat != 0)[0]
            for idx in mask_idx:
                classification_flat[idx] = pred_flat[idx] + 1

            # 4) 重塑回 (height, width)，并绘图
            classification_map = classification_flat.reshape((height, width))
            Draw_Classification_Map(classification_map,
                                    "results\\" + dataset_name + str(testOA))

            testing_time = toc2 - tic2 + LDA_SLIC_Time  # 分割耗时需要算进去
            Test_Time_ALL.append(testing_time)
            ## Saving data
            # sio.savemat(dataset_name+"softmax",{'softmax':output.reshape([height,width,-1]).cpu().numpy()})
            # np.save(dataset_name+"A_1", A_1.cpu().numpy())
            # np.save(dataset_name+"A_2", A_2.cpu().numpy())

        torch.cuda.empty_cache()
        del net

    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    AVG_ALL = np.array(AVG_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    print("\n==============================================================================")
    print('OA=', OA_ALL)
    print('AA=', AA_ALL)
    print('kppa=', KPP_ALL)
    print('AVG_ALL=', AVG_ALL)

    print('OA=', np.mean(OA_ALL), '+-', np.std(OA_ALL))
    print('AA=', np.mean(AA_ALL), '+-', np.std(AA_ALL))
    print('Kpp=', np.mean(KPP_ALL), '+-', np.std(KPP_ALL))
    print('AVG=', np.mean(AVG_ALL, 0), '+-', np.std(AVG_ALL, 0))
    print("Average training time:{}".format(np.mean(Train_Time_ALL)))
    print("Average testing time:{}".format(np.mean(Test_Time_ALL)))

    # 保存数据信息
    f = open('results\\' + dataset_name + '_results.txt', 'a+')
    str_results = '\n\n************************************************' \
                  + '\nOA=' + str(np.mean(OA_ALL)) + '+-' + str(np.std(OA_ALL)) \
                  + '\nAA=' + str(np.mean(AA_ALL)) + '+-' + str(np.std(AA_ALL)) \
                  + '\nKpp=' + str(np.mean(KPP_ALL)) + '+-' + str(np.std(KPP_ALL)) \
                  + '\nAVG=' + str(np.mean(AVG_ALL, 0)) + '+-' + str(np.std(AVG_ALL, 0)) \
                  + "\nAverage training time:{}".format(np.mean(Train_Time_ALL)) \
                  + "\nAverage testing time:{}".format(np.mean(Test_Time_ALL))
    f.write(str_results)
    f.close()







