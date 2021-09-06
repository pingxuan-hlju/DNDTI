import argparse
from my_function import *
from matrix_factorization import *
from model import *
import torch.utils.data as Data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Fold', type=int, default=5)
    parser.add_argument('--drug_num', type=int, default=708,
                        help='the number of drugs')
    parser.add_argument('--protein_num', type=int, default=1512,
                        help='the number of proteins')
    parser.add_argument('--disease_num', type=int, default=5603,
                        help='the number of diseases')
    parser.add_argument('--nmf_dim', type=int, default=200,
                        help='dimensions of low dimensional features')
    parser.add_argument('--iteration_num', type=int, default=40,
                        help='the maximum number of iterations')
    parser.add_argument('--alpha_1', type=float, default=0.1)
    parser.add_argument('--alpha_2', type=float, default=0.01)
    parser.add_argument('--lambda_', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate of dilation convolution')
    parser.add_argument('--cnn_lr', type=float, default=0.001,
                        help='learning rate of CNN')
    parser.add_argument('--EPOCH', type=int, default=50,
                        help='epoch of dilation convolution')
    parser.add_argument('--EPOCH_cnn', type=int, default=50,
                        help='epoch of CNN')
    parser.add_argument('--BATCH_SIZE', type=int, default=32,
                        help='BATCH_SIZE')

    args = parser.parse_args()
    print(args)
    Fold = args.Fold
    drug_num = args.drug_num
    protein_num = args.protein_num
    disease_num = args.disease_num
    nmf_dim = args.nmf_dim
    iteration_num = args.iteration_num
    alpha_1 = args.alpha_1
    alpha_2 = args.alpha_2
    lambda_ = args.lambda_
    lr = args.lr
    cnn_lr = args.cnn_lr
    EPOCH = args.EPOCH
    EPOCH_cnn = args.EPOCH_cnn
    BATCH_SIZE = args.BATCH_SIZE

    X_L = generate_X_L(A_rp, A_pr, S_rr_1, S_rr_2, S_rr_3, S_rr_4, S_pp_1, S_pp_2, S_pp_3, S_pp_4)
    G_r_1, G_r_2, G_r_3, G_r_4, G_p_1, G_p_2, G_p_3, G_p_4 = nmf(drug_num, protein_num, disease_num, A_rp, A_rd,A_pd, S_rr_1, S_rr_2, S_rr_3,
                               S_rr_4, S_pp_1, S_pp_2, S_pp_3, S_pp_4, S_dd_1, S_dd_2, nmf_dim, alpha_1,
                               alpha_2, iteration_num)
    X_R = generate_X_R(G_r_1, G_r_2, G_r_3, G_r_4, G_p_1, G_p_2, G_p_3, G_p_4)

    dilated_convolution = DilatedConvolution()
    CNN = CnnModule()
    attention = RelationAttention()
    if torch.cuda.is_available():
        dilated_convolution = dilated_convolution.cuda()
        CNN = CNN.cuda()
        attention = attention.cuda()

    optimizer = torch.optim.Adam([{'params': dilated_convolution.parameters(), 'lr': lr},
                                  {'params': CNN.parameters()},
                                  {'params': attention.parameters()}], lr=cnn_lr)
    loss_func_dc = nn.CrossEntropyLoss()
    loss_fun_cnn = nn.CrossEntropyLoss()

    train_data_set = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            x_dc = get_embedding_matrix_dc(x, X_L, drug_num)
            x_cnn = get_embedding_matrix_cnn(x, X_R, drug_num)
            x_dc = torch.from_numpy(x_dc).type(torch.FloatTensor)
            x_cnn = torch.from_numpy(x_cnn).type(torch.FloatTensor)
            y = y.type(torch.LongTensor)
            if torch.cuda.is_available():
                x_dc = x_dc.cuda()
                x_cnn = x_cnn.cuda()
                y = y.cuda()
            out_dc = dilated_convolution(x_dc)
            loss_dc = loss_func_dc(out_dc, y)
            att_cnn = attention(x_cnn)
            out_cnn = CNN(att_cnn)
            out = lambda_ * out_dc + (1 - lambda_) * out_cnn
            loss_all = loss_func_dc(out, y)
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            if step % 50 == 0:
                prediction_y = torch.max(out, 1)[1]
                accuracy = float((prediction_y == y).sum()) / float(y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss_all.data.cpu().numpy(),
                      '| test accuracy: %.2f' % accuracy)

    test_data_set = Data.TensorDataset(test_data, test_label)
    test_loader = Data.DataLoader(test_data_set, batch_size=BATCH_SIZE, shuffle=False)

    prediction_value = np.zeros((0, 2))

    for test_step, (test_x, test_y) in enumerate(test_loader):
        test_x_dc = get_embedding_matrix_dc(test_x, X_L, drug_num)
        test_x_cnn = get_embedding_matrix_cnn(test_x, X_R, drug_num)
        test_x_dc = torch.from_numpy(test_x_dc).type(torch.FloatTensor)
        test_x_cnn = torch.from_numpy(test_x_cnn).type(torch.FloatTensor)
        test_y = test_y.type(torch.LongTensor)
        if torch.cuda.is_available():
            test_x_dc = test_x_dc.cuda()
            test_x_cnn = test_x_cnn.cuda()
            test_y = test_y.cuda()
        test_out_dc = dilated_convolution(test_x_dc)
        test_att_cnn = attention(test_x_cnn)
        test_out_cnn = CNN(test_att_cnn)
        test_out = lambda_ * test_out_dc + (1 - lambda_) * test_out_cnn
        test_out = F.softmax(test_out, dim=1)
        prediction_value = np.vstack((prediction_value, test_out.detach().cpu().numpy()))
