import numpy as np
from copy import deepcopy
import sys

min_real = sys.float_info.min


def generate_matrix_list(matrix_1, matrix_2, matrix_3=None, matrix_4=None, is_disease=False):
    matrix_list = []
    if is_disease:
        matrix_list.append(matrix_1)
        matrix_list.append(matrix_2)
    else:
        matrix_list.append(matrix_1)
        matrix_list.append(matrix_2)
        matrix_list.append(matrix_3)
        matrix_list.append(matrix_4)
    return matrix_list


def update_G_r(G_r, G_p, G_d, S_rr, A_rp, A_rd, W_rp, W_rd, alpha_1):
    G_r_new = []
    for phi_r in range(len(G_r)):
        G = deepcopy(G_r[phi_r])
        S = deepcopy(S_rr[phi_r])

        # numerator
        formula_1 = 2 * S.dot(G)
        formula_2 = 0
        formula_3 = 0
        for phi_p in range(len(G_p)):
            formula_2_temp = alpha_1 * A_rp.dot(G_p[phi_p]).dot(np.transpose(W_rp))
            formula_2 = formula_2 + formula_2_temp
        for phi_d in range(len(G_d)):
            formula_3_temp = alpha_1 * A_rd.dot(G_d[phi_d]).dot(np.transpose(W_rd))
            formula_3 = formula_3 + formula_3_temp
        numerator = formula_1 + formula_2 + formula_3

        # denominator
        d_formula_1 = 2 * G.dot(np.transpose(G)).dot(G)
        d_formula_2 = 0
        d_formula_3 = 0
        for phi_p in range(len(G_p)):
            d_formula_2_temp = alpha_1 * G.dot(W_rp).dot(np.transpose(G_p[phi_p])).dot(G_p[phi_p]).dot(np.transpose(W_rp))
            d_formula_2 = d_formula_2 + d_formula_2_temp
        for phi_d in range(len(G_d)):
            d_formula_3_temp = alpha_1 * G.dot(W_rd).dot(np.transpose(G_d[phi_d])).dot(G_d[phi_d]).dot(np.transpose(W_rd))
            d_formula_3 = d_formula_3 + d_formula_3_temp
        denominator = d_formula_1 + d_formula_2 + d_formula_3
        G_new = np.multiply(np.divide(numerator, denominator), G)
        G_new = np.maximum(G_new, min_real)
        G_r_new.append(G_new)
    return G_r_new


def update_G_p(G_p, G_r, G_d, S_pp, A_rp, A_pd, W_rp, W_pd, alpha_1):
    G_p_new = []
    for phi_p in range(len(G_p)):
        G = deepcopy(G_p[phi_p])
        S = deepcopy(S_pp[phi_p])

        # numerator
        formula_1 = 2 * S.dot(G)
        formula_2 = 0
        formula_3 = 0
        for phi_r in range(len(G_r)):
            formula_2_temp = alpha_1 * np.transpose(A_rp).dot(G_r[phi_r]).dot(W_rp)
            formula_2 = formula_2 + formula_2_temp
        for phi_d in range(len(G_d)):
            formula_3_temp = alpha_1 * A_pd.dot(G_d[phi_d]).dot(np.transpose(W_pd))
            formula_3 = formula_3 + formula_3_temp
        numerator = formula_1 + formula_2 + formula_3

        # denominator
        d_formula_1 = 2 * G.dot(np.transpose(G)).dot(G)
        d_formula_2 = 0
        d_formula_3 = 0
        for phi_r in range(len(G_r)):
            d_formula_2_temp = alpha_1 * G.dot(np.transpose(W_rp)).dot(np.transpose(G_r[phi_r])).dot(G_r[phi_r]).dot(
                W_rp)
            d_formula_2 = d_formula_2 + d_formula_2_temp
        for phi_d in range(len(G_d)):
            d_formula_3_temp = alpha_1 * G.dot(W_pd).dot(np.transpose(G_d[phi_d])).dot(G_d[phi_d]).dot(
                np.transpose(W_pd))
            d_formula_3 = d_formula_3 + d_formula_3_temp
        denominator = d_formula_1 + d_formula_2 + d_formula_3
        G_new = np.multiply(np.divide(numerator, denominator), G)
        G_new = np.maximum(G_new, min_real)
        G_p_new.append(G_new)
    return G_p_new


def update_G_d(G_d, G_r, G_p, S_dd, A_rd, A_pd, W_rd, W_pd, alpha_1):
    G_d_new = []
    for phi_d in range(len(G_d)):
        G = deepcopy(G_d[phi_d])
        S = deepcopy(S_dd[phi_d])

        # numerator
        formula_1 = 2 * S.dot(G)
        formula_2 = 0
        formula_3 = 0
        for phi_r in range(len(G_r)):
            formula_2_temp = alpha_1 * np.transpose(A_rd).dot(G_r[phi_r]).dot(W_rd)
            formula_2 = formula_2 + formula_2_temp
        for phi_p in range(len(G_p)):
            formula_3_temp = alpha_1 * np.transpose(A_pd).dot(G_p[phi_p]).dot(W_pd)
            formula_3 = formula_3 + formula_3_temp
        numerator = formula_1 + formula_2 + formula_3

        # denominator
        d_formula_1 = 2 * G.dot(np.transpose(G)).dot(G)
        d_formula_2 = 0
        d_formula_3 = 0
        for phi_r in range(len(G_r)):
            d_formula_2_temp = alpha_1 * G.dot(np.transpose(W_rd)).dot(np.transpose(G_r[phi_r])).dot(G_r[phi_r]).dot(
                W_rd)
            d_formula_2 = d_formula_2 + d_formula_2_temp
        for phi_p in range(len(G_p)):
            d_formula_3_temp = alpha_1 * G.dot(np.transpose(W_pd)).dot(np.transpose(G_p[phi_p])).dot(G_p[phi_p]).dot(
                W_pd)
            d_formula_3 = d_formula_3 + d_formula_3_temp
        denominator = d_formula_1 + d_formula_2 + d_formula_3
        G_new = np.multiply(np.divide(numerator, denominator), G)
        G_new = np.maximum(G_new, min_real)
        G_d_new.append(G_new)
    return G_d_new


def update_W_rp(W_rp, A_rp, G_r, G_p, alpha_1, alpha_2):
    numerator = 0
    denominator = 0
    # numerator
    for phi_r in range(len(G_r)):
        for phi_p in range(len(G_p)):
            formula = alpha_1 * np.transpose(G_r[phi_r]).dot(A_rp).dot(G_p[phi_p])
            numerator = numerator + formula
    # denominator
    for phi_r in range(len(G_r)):
        for phi_p in range(len(G_p)):
            d_formula = alpha_1 * np.transpose(G_r[phi_r]).dot(G_r[phi_r]).dot(W_rp).dot(np.transpose(G_p[phi_p])).dot(G_p[phi_p])
            denominator = denominator + d_formula
    denominator = denominator + alpha_2 * W_rp
    W_rp_new = np.multiply(np.divide(numerator, denominator), W_rp)
    W_rp_new = np.maximum(W_rp_new, min_real)
    return W_rp_new


def update_W_rd(W_rd, A_rd, G_r, G_d, alpha_1, alpha_2):
    numerator = 0
    denominator = 0
    # numerator
    for phi_r in range(len(G_r)):
        for phi_d in range(len(G_d)):
            formula = alpha_1 * np.transpose(G_r[phi_r]).dot(A_rd).dot(G_d[phi_d])
            numerator = numerator + formula
    # denominator
    for phi_r in range(len(G_r)):
        for phi_d in range(len(G_d)):
            d_formula = alpha_1 * np.transpose(G_r[phi_r]).dot(G_r[phi_r]).dot(W_rd).dot(np.transpose(G_d[phi_d])).dot(G_d[phi_d])
            denominator = denominator + d_formula
    denominator = denominator + alpha_2 * W_rd
    W_rd_new = np.multiply(np.divide(numerator, denominator), W_rd)
    W_rd_new = np.maximum(W_rd_new, min_real)
    return W_rd_new


def update_W_pd(W_pd, A_pd, G_p, G_d, alpha_1, alpha_2):
    numerator = 0
    denominator = 0
    # numerator
    for phi_p in range(len(G_p)):
        for phi_d in range(len(G_d)):
            formula = alpha_1 * np.transpose(G_p[phi_p]).dot(A_pd).dot(G_d[phi_d])
            numerator = numerator + formula
    # denominator
    for phi_p in range(len(G_p)):
        for phi_d in range(len(G_d)):
            d_formula = alpha_1 * np.transpose(G_p[phi_p]).dot(G_p[phi_p]).dot(W_pd).dot(np.transpose(G_d[phi_d])).dot(G_d[phi_d])
            denominator = denominator + d_formula
    denominator = denominator + alpha_2 * W_pd
    W_pd_new = np.multiply(np.divide(numerator, denominator), W_pd)
    W_pd_new = np.maximum(W_pd_new, min_real)
    return W_pd_new


def error_func(G_r, G_p, G_r_new, G_p_new):
    loss = 0
    for phi_r in range(len(G_r)):
        loss = np.sum(np.abs(G_r[phi_r] - G_r_new[phi_r])) + loss
    for phi_p in range(len(G_p)):
        loss = np.sum(np.abs(G_p[phi_p] - G_p_new[phi_p])) + loss
    return loss


def iterative_update(A_rp, A_rd, A_pd, S_rr, S_pp, S_dd, G_r, G_p, G_d, W_rp, W_rd, W_pd, alpha_1,
                     alpha_2, iteration_num):
    for i in range(iteration_num):

        W_rp_new = update_W_rp(W_rp, A_rp, G_r, G_p, alpha_1, alpha_2)
        W_rd_new = update_W_rd(W_rd, A_rd, G_r, G_d, alpha_1, alpha_2)
        W_pd_new = update_W_pd(W_pd, A_pd, G_p, G_d, alpha_1, alpha_2)
        W_rp = W_rp_new
        W_rd = W_rd_new
        W_pd = W_pd_new

        G_r_new = update_G_r(G_r, G_p, G_d, S_rr, A_rp, A_rd, W_rp, W_rd, alpha_1)
        G_p_new = update_G_p(G_p, G_r, G_d, S_pp, A_rp, A_pd, W_rp, W_pd, alpha_1)
        G_d_new = update_G_d(G_d, G_r, G_p, S_dd, A_rd, A_pd, W_rd, W_pd, alpha_1)

        error = error_func(G_r, G_p, G_r_new, G_p_new)
        G_r = G_r_new
        G_p =G_p_new
        G_d = G_d_new
        if error < 10e-6:
            break

    return G_r, G_p


def nmf(drug_num, protein_num, disease_num, A_rp, A_rd, A_pd, S_rr_1, S_rr_2, S_rr_3, S_rr_4, S_pp_1, S_pp_2,
        S_pp_3, S_pp_4, S_dd_1, S_dd_2, nmf_dim, alpha_1, alpha_2, iteration_num):
    W_rp = np.random.randn(nmf_dim, nmf_dim)
    W_rd = np.random.randn(nmf_dim, nmf_dim)
    W_pd = np.random.randn(nmf_dim, nmf_dim)
    G_r_1 = np.random.randn(drug_num, nmf_dim)
    G_r_2 = np.random.randn(drug_num, nmf_dim)
    G_r_3 = np.random.randn(drug_num, nmf_dim)
    G_r_4 = np.random.randn(drug_num, nmf_dim)
    G_p_1 = np.random.randn(protein_num, nmf_dim)
    G_p_2 = np.random.randn(protein_num, nmf_dim)
    G_p_3 = np.random.randn(protein_num, nmf_dim)
    G_p_4 = np.random.randn(protein_num, nmf_dim)
    G_d_1 = np.random.randn(disease_num, nmf_dim)
    G_d_2 = np.random.randn(disease_num, nmf_dim)

    S_rr = generate_matrix_list(S_rr_1, S_rr_2, S_rr_3, S_rr_4)
    S_pp = generate_matrix_list(S_pp_1, S_pp_2, S_pp_3, S_pp_4)
    S_dd = generate_matrix_list(S_dd_1, S_dd_2, None, None, True)

    G_r = generate_matrix_list(G_r_1, G_r_2, G_r_3, G_r_4)
    G_p = generate_matrix_list(G_p_1, G_p_2, G_p_3, G_p_4)
    G_d = generate_matrix_list(G_d_1, G_d_2, None, None, True)

    G_r, G_p = iterative_update(A_rp, A_rd, A_pd, S_rr, S_pp, S_dd, G_r, G_p, G_d, W_rp, W_rd, W_pd, alpha_1,
                                alpha_2, iteration_num)
    G_r_1 = G_r[0]
    G_r_2 = G_r[1]
    G_r_3 = G_r[2]
    G_r_4 = G_r[3]
    G_p_1 = G_p[0]
    G_p_2 = G_p[1]
    G_p_3 = G_p[2]
    G_p_4 = G_p[3]

    return G_r_1, G_r_2, G_r_3, G_r_4, G_p_1, G_p_2, G_p_3, G_p_4
