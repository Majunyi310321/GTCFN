# GTCFN

GTCFN: A Graph-based Transformer and Convolution Fusion Network for Hyperspectral Image Classification

has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS)

Abstract: —Graph Neural Networks (GNN) are capable of modeling complex non-Euclidean structures through information transfer, and thus have been party widely used in the field of
 Hyperspectral Image (HSI) classification. However, conventional GNNs often have difficulty in handling regular grid data, which in turn loses positional information or spatial coherence, as well
 as in capturing long-range dependencies, which affects their performance in heterogeneous and limited-sample condition. To address these limitations, this paper proposes a novel Graph
based Transformer and Convolution Fusion Network (GTCFN) that integrates the local representation power of Convolutional Neural Networks (CNNs) with the global reasoning capability
 of graph-based Transformers. GTCFN consists of two synergistic branches: a Graph Transformer sub-network (GTsN) that models high-level semantic structures among superpixels
 via attention-based topology learning, and a Spectral–Spatial Convolutional sub-network (S2CsN) that extracts multi-scale finegrained features using 5×5, 7×7, and 9×9 convolutional kernels.
 To enhance efficiency and generalization, GTCFN incorporates kernelized attention with random feature mapping, reducing the complexity from O(M2) to O(M). At the same time, attention oversmoothing is avoided by introducing a Gumbel-based multi
head random aggregation mechanism. Experiments conducted on four benchmark datasets, namely Indian Pines, Pavia University, Salinas and WHU-Hi-HongHu, show that GTCFN achieves stateof-the-art performance with OA of 95.62%, 98.34%, 97.88% and
 96.69%, which is significantly better than 12 other algorithms, such as CNNs, graph-based models and hybrid network models.

Environment: 
Python 3.7.4
PyTorch 1.8.1


If this work is helpful to you, please citing our work as follows: 

 X. Zhao, J. Ma, L. Wang, "GTCFN: A Graph-based Transformer and Convolution Fusion Network for Hyperspectral Image Classification" in IEEE Transactions on Geoscience and Remote Sensing,

Note that the code is stinky and long.
