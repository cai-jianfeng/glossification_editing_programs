# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: viterbi.py
@Date: 2023/11/10 22:37
@Author: caijianfeng
"""
import numpy as np
import copy

def viterbi(matrixs, paths, dist, cur):
    """
    :param matrixs: list(numpy.array) 列表中的第 i 个矩阵表示第 i 列的节点到第 i+1 列的节点的连接矩阵
    :param paths: list(list) 表示从起点到第 cur 列节点的所有最短路径(数量 = 第 cur 列节点的数量)
    :param dist: list 表示从起点到第 cur 列节点的所有最短路径距离(数量 = 第 cur 列节点的数量)
    :param cur: int 表示当前遍历的节点的列数
    :return: 从起点到终点的最短路径
    """
    if cur >= len(matrixs):
        return paths[0]
    matrix = matrixs[cur]
    if not paths and not dist:  # 表示初始节点/起点
        for j in range(matrix.shape[1]):
            dist.append(matrix[0][j])  # 记录起点到第一排节点的所有距离
            paths.append([j])
        path = viterbi(matrixs, paths, dist, cur+1)
    else:
        n_paths = []
        n_dist = []
        for j in range(matrix.shape[1]):  # 第 cur + 1 排点
            temp = dist[0] + matrix[0][j]
            temp_v = 0
            for k in range(matrix.shape[0]):  # 第 cur 排点
                if dist[k] + matrix[k][j] < temp:
                    temp = dist[k] + matrix[k][j]
                    temp_v = k
            n_paths.append(paths[temp_v] + [j])
            n_dist.append(temp)
        paths = n_paths
        dist = n_dist
        path = viterbi(matrixs, paths, dist, cur+1)
    return path


def matrix2graph(matrix):
    """
    将概率矩阵转化为图矩阵
    :param matrix: 概率矩阵
    :return: 图矩阵
    """
    graph = []
    for i in range(matrix.shape[1]):
        sub_graph = np.tile(matrix[:, i], (matrix.shape[0], 1))
        graph.append(sub_graph)
    return graph


def viterbi_matrix_dfs(matrix, paths, path, p, cur, high):
    """
    :param matrix: np.array  概率矩阵，行表示每个 word token，列表示每个 video clip embedding
    :param path: list 当前路径
    :param paths: list(tuple(float, list)) 所有可行路径
    :param p: float 当前路径的累积概率
    :param cur: int 当前遍历的列数
    :return: None
    """
    if cur >= matrix.shape[1]:
        if path[-1] == matrix.shape[0]-1:
            paths.append((p, path))
        return
    if high < matrix.shape[0]:
        n_path = copy.deepcopy(path)  # python 的 list 传参是浅复制
        n_path.append(high)
        p = p + matrix[high, cur]
        viterbi_matrix_dfs(matrix, paths, n_path, p, cur+1, high)
        viterbi_matrix_dfs(matrix, paths, n_path, p, cur+1, high+1)


def viterbi_matrix(matrix):
    """
    :param matrix: np.array 概率矩阵
    :return: list, list(list) 表示最短路径 和 所有可行路径集合
    """
    cur, high = 0, 0
    paths, path, p = [], [], 0.0
    viterbi_matrix_dfs(matrix, paths, path, p, cur, high)  # paths 中存储所有可行路径
    temp = paths[0]
    for pa in paths:
        temp = pa if pa > temp else temp
    return temp, paths


if __name__ == '__main__':
    # 测试普通的 viterbi 算法
    matrix_1 = np.array([[2, 3, 7]])  # (1, 3)
    matrix_2 = np.array([[4, 5, 6], [10, 3, 9], [12, 8, 5]])  # (3, 3)
    matrix_3 = np.array([[5, 6, 10], [4, 7, 2], [3, 5, 7]])  # (3, 3)
    matrix_4 = np.array([[1], [9], [3]])  # (3, 1)
    matrixs = [matrix_1, matrix_2, matrix_3, matrix_4]

    paths = []
    dist = []

    path = viterbi(matrixs, paths, dist, cur=0)

    print("path: ", end="")
    for p in path:
        print(p, end=" ")
    print()

    # 测试改进的 viterbi 算法
    matrix = np.random.random([3, 5])

    (p, path), paths = viterbi_matrix(matrix)

    print(matrix)
    print(f"propbility: {p}; path: ", end="")
    for p in path:
        print(p, end=" ")
