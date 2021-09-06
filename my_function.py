import numpy as np


def generate_S_list(matrix_1, matrix_2, matrix_3, matrix_4):
    matrix_list = list()
    matrix_list.append(matrix_1)
    matrix_list.append(matrix_2)
    matrix_list.append(matrix_3)
    matrix_list.append(matrix_4)
    return matrix_list


def generate_X_L(A_rp, A_pr, S_rr_1, S_rr_2, S_rr_3, S_rr_4, S_pp_1, S_pp_2, S_pp_3, S_pp_4):
    S_rr = np.array(generate_S_list(S_rr_1, S_rr_2, S_rr_3, S_rr_4))
    S_pp = np.array(generate_S_list(S_pp_1, S_pp_2, S_pp_3, S_pp_4))
    X_L = []
    for phi_r in range(len(S_rr)):
        for phi_p in range(len(S_pp)):
            R = S_rr[phi_r]
            P = S_pp[phi_p]
            R = np.hstack((R, A_rp))
            P = np.hstack((A_pr, P))
            R_P = np.vstack((R, P))
            X_L.append(R_P)
    X_L = np.array(X_L)
    return X_L


def feature_concatenation(matrix_1, matrix_2, matrix_3, matrix_4):
    feaure_matrix = np.hstack((matrix_1, matrix_2))
    feaure_matrix = np.hstack((feaure_matrix, matrix_3))
    feaure_matrix = np.hstack((feaure_matrix, matrix_4))
    return feaure_matrix


def generate_X_R(G_r_1, G_r_2, G_r_3, G_r_4, G_p_1, G_p_2, G_p_3, G_p_4):
    G_r_feature = feature_concatenation(G_r_1, G_r_2, G_r_3, G_r_4)
    G_p_feature = feature_concatenation(G_p_1, G_p_2, G_p_3, G_p_4)
    X_R = np.vstack((G_r_feature, G_p_feature))
    return X_R


def get_embedding_matrix_dc(pairwise, X_L, drug_num):
    batch_size = len(pairwise)
    pairwise_embedding_list = list()
    for i in range(batch_size):
        x, y = pairwise[i][0], pairwise[i][1]
        pairwise_embedding = []
        for n in range(len(X_L)):
            r = X_L[n][x]
            p = X_L[n][y + drug_num]
            rp = np.vstack((r, p))
            pairwise_embedding.append(rp)
        pairwise_embedding_list.append(pairwise_embedding)
    pairwise_embedding_list = np.array(pairwise_embedding_list)
    return pairwise_embedding_list


def get_embedding_matrix_cnn(pairwise, X_R, drug_num):
    batch_size = len(pairwise)
    pairwise_embedding_list = list()
    for i in range(batch_size):
        x, y = pairwise[i][0], pairwise[i][1]
        r = X_R[x]
        p = X_R[y + drug_num]
        rp = np.hstack((r, p)).reshape(8, 200)
        pairwise_embedding_list.append([rp])
    pairwise_embedding_list = np.array(pairwise_embedding_list)
    return pairwise_embedding_list

