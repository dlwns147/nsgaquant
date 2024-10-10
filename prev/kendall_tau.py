from acc_predictor.factory import get_acc_predictor
import json
import numpy as np

# import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# train_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_loss_uniform_1000.json'
# test_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_loss_uniform_test.json'
# # train_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_1000.json'
# # test_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_test.json'
# predictor_type = 'mlp'

# with open(train_dataset_path, 'r') as json_file:
#     train_dataset = json.load(json_file)
# with open(test_dataset_path, 'r') as json_file:
#     test_dataset = json.load(json_file)


# # x = np.array(list(train_dataset.keys()))
# train_x = np.array([np.fromstring(k, sep=' ') for k in train_dataset.keys()])
# train_y = np.array(list(train_dataset.values()))

# test_x = np.array([np.fromstring(k, sep=' ') for k in test_dataset.keys()])
# test_y = np.array(list(test_dataset.values()))
# acc_predictor = get_acc_predictor(predictor_type, train_x, train_y)

# print(f'===== train set ======')
# train_error_list = []
# train_out_list = []
# for data, label in zip(train_x, train_y):
#     out = acc_predictor.predict(data).item()
#     print(f'label : {label:.3f}, out : {out:.3f}')
#     error = abs(label - out)
#     train_out_list.append(out)
#     train_error_list.append(error)

# train_r2 = r2_score(train_y, train_out_list)

# test_error_list = []
# test_out_list = []

# print(f'===== test set ======')
# for data, label in zip(test_x, test_y):
#     out = acc_predictor.predict(data).item()
#     print(f'label : {label:.3f}, out : {out:.3f}')
#     error = abs(label - out)
#     test_out_list.append(out)
#     test_error_list.append(error)

# test_r2 = r2_score(test_y, test_out_list)
# print(f'train error : {sum(train_error_list):.3f}, mean : {np.mean(train_error_list):.3f}, std : {np.std(train_error_list):.3f}, r2 : {train_r2:.2f}')
# print(f'test error : {sum(test_error_list):.3f}, mean : {np.mean(test_error_list):.3f}, std : {np.std(test_error_list):.3f}, r2 : {test_r2:.2f}')

from scipy.stats import kendalltau, spearmanr
test_loss_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_loss_uniform_test.json'
test_ppl_dataset_path = '/NAS/SJ/hqq/arch/Llama-2-7b-hf_ppl_uniform_test.json'
with open(test_loss_dataset_path, 'r') as json_file:
    test_loss_dataset = json.load(json_file)
with open(test_ppl_dataset_path, 'r') as json_file:
    test_ppl_dataset = json.load(json_file)
test_loss = np.array(list(test_loss_dataset.values()))
test_ppl = np.array(list(test_ppl_dataset.values()))

print(kendalltau(np.argsort(test_loss), np.argsort(test_ppl)))
print(spearmanr(np.argsort(test_loss), np.argsort(test_ppl)))
exit()
# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# ax[0].scatter(train_out_list, train_y, s=5)
# ax[0].set_title('Train Dataset')
# ax[0].set_xlabel('Predicted Output')
# ax[0].set_ylabel('Real Output')
# ax[0].text(0.5, 0.5, f'R^2: {train_r2}', fontsize=12)
# ax[1].scatter(test_out_list, test_y, s=5)
# ax[1].set_title('Test Dataset')
# ax[1].set_xlabel('Predicted Output')
# ax[1].set_ylabel('Real Output')
# ax[1].text(0.5, 0.5, f'R^2: {test_r2}', fontsize=12)
# plt.show()
# plt.savefig('test.png', dpi=300)

# print(f'train_error_list : {train_error_list}')
# print(f'test_error_list : {test_error_list}')
# print(f'train error : {sum(train_error_list)}, mean : {(sum(train_error_list) / len(train_error_list))}')
# print(f'test error : {sum(test_error_list)}, mean : {(sum(test_error_list) / len(test_error_list))}')

