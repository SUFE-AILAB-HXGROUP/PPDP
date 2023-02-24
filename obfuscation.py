import numpy as np
import scipy.spatial.distance as dist
import scipy.io
import matlab.engine

from tqdm import tqdm
from sklearn.preprocessing import normalize


def get_DP_obf_X(df, dist_matrix, beta):
    """
    :param df: data to be obfuscated
    :param dist_matrix: distance matrix between users
    :param beta: parameter of differential privacy
    :return: X_obf - obfuscated data, X_ori - original data
    """
    X_ori = {}
    X_obf = {}
    user_size = df.shape[0]
    uid_list = list(df['uid'].values)
    for i in range(user_size):
        user_id = df['uid'][i]
        # get X_ori
        X_ori[user_id] = df[df['uid'] == user_id].values[0, :]

        # get X_obf
        dist_arr = np.array(list(dist_matrix[i]))
        dp_arr = np.exp(-beta*dist_arr)
        prob_list = list(dp_arr / sum(dp_arr))   # compute the swap probabilities
        uidx = np.random.choice(uid_list, 1, p=prob_list)[0]   # randomly choose a user for swap
        X_obf[user_id] = df[df['uid'] == uidx].values[0, :]

    return X_obf, X_ori


def differential_privacy(df_test, beta, repeats=100):
    print("Obfuscation method: DP, beta: {}".format(beta))
    print("generate distance matrix...")
    dist_mat = dist.squareform(dist.pdist(df_test.iloc[:, :-2], 'jaccard'))

    print("start obfuscating...")
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_DP_obf_X(df_test, dist_mat, beta)
    _, X_ori = get_DP_obf_X(df_test, dist_mat, beta)
    print("obfuscating done.")

    return X_obf_dict, X_ori


def get_random_obf_X(df, p_rand):
    X_ori = {}
    X_obf = {}

    user_size = df.shape[0]
    uid_list = list(df['uid'].values)

    for i in range(user_size):
        obf_flag = np.random.choice([0, 1], 1, p=[0, 1])
        user_id = df['uid'][i]

        # get X_ori
        X_ori[user_id] = df[df['uid']==user_id].values[0, :]

        if obf_flag == 1:
            # get X_obf
            flag = np.random.choice([0, 1], 1, p=[1-p_rand, p_rand])[0]
            if flag == 0:
                X_obf[user_id] = df[df['uid']==user_id].values[0, :]
            else:
                ul = [user_id]
                uidx = np.random.choice(list(set(uid_list) - set(ul)), 1)[0]
                X_obf[user_id] = df[df['uid']==uidx].values[0, :]
        else:
            X_obf[user_id] = df[df['uid']==user_id].values[0, :]

    return X_obf, X_ori


def random_obf(df_test, p_rand, repeats=100):
    print("Obfuscation method: Random, p_rand: {}".format(p_rand))
    print("start obfuscating...")
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_random_obf_X(df_test, p_rand)
    _, X_ori = get_random_obf_X(df_test, p_rand)
    print("obfuscating done.")

    return X_obf_dict, X_ori


def get_obf_X(df, xpgg):
    user_size = df.shape[0]
    X_obf = {}
    X_ori = {}
    xpgg[xpgg < 0.00001] = 0
    xpgg_norm = normalize(xpgg, axis=0, norm='l1')
    for i in range(user_size):
        user_id = df['uid'][i]
        X_ori[user_id] = df[df['uid'] == user_id].values[0, :]

        # selecting one cluster to change
        uidx = np.random.choice()
        X_obf[user_id] = df[df['uid'] == uidx].values[0, :]


        #     # selecting one cluster to change
        #     while True:
        #         change_index = np.random.choice(range(0, cluster_size * ageGroup_size), 1, p=xpgg_norm[:,
        #                                                                                      user_ageGroup_dict[user_id] + (
        #                                                                                                  user_cluster_dict[
        #                                                                                                      user_id] - 1) * ageGroup_size])[
        #             0]
        #         change_cluster_index = int(change_index / ageGroup_size) + 1
        #         change_ageGroup_index = change_index % ageGroup_size
        #         potential_users = list(set(cluster_vec[change_cluster_index]) & set(ageGroup_vec[change_ageGroup_index]))
        #         if len(potential_users) > 0: # potential users may be empty by a slight probability
        #             break
        #         else:
        #             print("not find potential users, re-pick")
        #     #         print(change_index, change_cluster_index, change_ageGroup_index, potential_users)
        #     uidx = np.random.choice(potential_users, 1)[0]
        #     X_obf[user_id] = df_cluster[df_cluster['uid'] == uidx].values[0, :-3]
        # else:
        #     X_obf[user_id] = df_cluster[df_cluster['uid']==user_id].values[0, :-3]

    return X_obf, X_ori


def PrivCheck(df_test, deltaX, repeats=100):
    print("Obfuscation method: PrivCheck, deltaX: {}".format(deltaX))
    # solve the obfuscation probability matrix
    pd.DataFrame(funcs.cal_pgy_withAgeGroup(df_test, cluster_num, 1, age_list)).to_csv(
        'tmp/pgy_ageGroup_privcheck.csv',
        index=False, header=None)

    JSD_Mat_dict = np.zeros((cluster_num, cluster_num, age_group_number))
    group_min_age_dict = {}
    group_usersize_dict = {}
    for ag in range(age_group_number):
        group_min_age_dict[ag] = group_age_dict[ag][0]
        df_test_ag = df_test.loc[df_test['age_group'] == ag]
        age_list_ag = group_age_dict[ag]
        group_usersize_dict[ag] = df_test_ag.shape[0]

        JSD_Mat_dict[:, :, ag] = funcs.cal_JSD_Matrix_withoutAgeGroup(df_test_ag, cluster_num, 4)
    scipy.io.savemat('tmp/JSDM_ageGroup_privcheck.mat', {"JSD_Mat_input": JSD_Mat_dict})

    pd.DataFrame(JSD_Mat_dict[ag]).to_csv('tmp/JSDM_ageGroup_yang.csv', index=False, header=None)

    eng = matlab.engine.start_matlab()
    eng.edit('../../matlab/age_tradeoff_scenario_I/PrivCheck', nargout=0)
    eng.cd('../../matlab/age_tradeoff_scenario_I', nargout=0)
    xpgg, distortion_budget = np.array(eng.PrivCheck(deltaX, nargout=2))
    xpgg = np.array(xpgg)

    # obfuscation
    X_obf_dict = {}
    for i in tqdm(range(repeats)):
        X_obf_dict[i], _ = get_obf_X(df_test, xpgg, pp)
    _, X_ori = get_obf_X(df_test, xpgg, pp)

    return X_obf_dict, X_ori