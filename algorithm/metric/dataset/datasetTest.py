from scipy.io import loadmat
from scipy.io import savemat


# # 指定MAT文件的路径
# mat_file_path = 'smd_machine1_1_train.mat'
#
# # 使用loadmat函数加载MAT文件
# mat_data = loadmat(mat_file_path)
#
# # 查看MAT文件中的变量和数据
# print("MAT文件中的变量:")
# print(mat_data.keys())
#
# # 获取特定变量的值
# var_name = 'machine1'
# var_value = mat_data[var_name]
# print("变量 {} 的值:".format(var_name))
# print(var_value[:100])

mat_file_path = 'real_data.mat'
mat_data = loadmat(mat_file_path)
mat_data['machine1'] = mat_data['machine1'][:100]
savemat(mat_file_path, mat_data)

