# 图神经网络 作业1

## Problem 1

(1)(2)(3)三道题对应的函数分别写在`nodeutils.py`、`edgeutils.py`、`graphutils.py`中，运行`python KarateFeat.py`在Karate数据集上调用这些函数并奖结果存储到`KarateResult.txt`文件中

(1) 度数计算：在边表中寻找所有源为该节点的边并统计个数

聚集系数计算：取出节点的所有邻居$N$，将边表转换为邻接矩阵$A$，遍历邻居节点对$(v_i,v_j)$，若$A_{ij}$非零则给邻居节点连边数加一。最后将邻居节点的连边数除以$|N|(|N|-1)/2$得到聚集系数

(2) Jaccard系数计算：对每对节点取出它们的邻居节点，用Python集合操作计算交并比

Katz系数计算：使用课件上的公式，将边表转换为邻接矩阵$A$并计算$S=\displaystyle\sum_{i=1}^{\infty}\beta^i A^i = (I - \beta A)^{-1}-I$

(3) 对于G1和G2，对结点的三元组统计度数之和，若为4，则G1数目加一，若为6，则G2数目加一

对于G3-G8，对结点的四元组计算每个节点的度数，根据这个度数向量判断graphlet类型，判断细节在`graphutils.py`中的注释中

## Problem 2

邻接矩阵的幂$A_{ij}^k$表示节点$ij$之间长为$k$的路径个数，一个节点要经过3条边回到本身，只可能是经过一个三角形的路径，因此$trace(A^3)$是图中有向三角形个数的3倍(所有三角形每个顶点均贡献了1，因此一个三角形的贡献为3)，也即无向三角形个数的6倍

## (Optional) Problem A

在`graphutils.py`中实现了快速计数graphlet的函数`count_graphlets_fast`，并使用`python CoraGraphlet.py`中调用该函数完成在Cora数据集上的计算，整体用时~2min，主要的思路都是借助邻接矩阵$A$，并除以对称性带来的系数

- G1：每个顶点$u$，则$u$与其任意两个邻居组成的子图或者为G1，或者为G2，统计$u$的邻居数目$N(u)$并计算$N(u)(N(u)-1)/2$，再减去G2三角形个数即可
- G2：用Problem 2的结论，计算$trace(A^3)/6$
- G3：对所有点对$u,v$，统计$(u,w_1,w_2,v)$路径的条数，其中$A_{uv}$、$A_{w_1 v}$、$A_{uw_2}$均为零
- G4：对所有点$u$，统计其邻居的互不相邻三元组数目
- G5：对所有路径为2的点对$u,v$，求出共同邻居$N$，统计$(w_1,w_2)\in N\times N$中使得$A_{w_1w_2}=0$的点对数目
- G6：对所有点$u$，求出邻居节点$N(u)$，对所有的$v\in N(u)$，求出$N(u)\setminus N(v)$中相邻的点对数
- G7：与G5类似，但应统计使得$A_{w_1w_2}=1$的点对数目
- G8：与G4类似，但统计邻居中构成三角形的数目

## (Optional) Problem B

(1) 两图中节点个数和所有节点的度都是相同的，因此在Color Refinement中每一轮每一节点都会得到相同的结果，因此Weisfeiler-Lehman核无法区分二者

(2) 两者的graphlet向量不同，代码在`DistinguishGraphs.py`中