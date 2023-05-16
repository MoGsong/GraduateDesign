import matplotlib.pyplot as plt
import numpy as np

# 计算每个类别的平均准确率
acc_mean = [0.8149, 0.8411, 0.7490, 0.8030]
# acc_mean = [0.7249, 0.7345, 0.5380, 0.7226]
# Kappa = [0.645, 0.514, 0.583]
# 绘制柱形图
fig, ax = plt.subplots()
rects = ax.bar(np.arange(4), acc_mean, align='center', color='b')
# 设置x轴标签
ax.set_xticks(np.arange(4))
ax.set_xticklabels(('MI-EEGNet+DA', 'CSP', 'ShallowNet'))
# 设置y轴标签和标题
ax.set_ylabel('Kappa')
ax.set_title('Kappa value for different method, 4')
# 在每个柱形图上添加标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.0*height,
                '%.4f' % height,
                ha='center', va='bottom')

autolabel(rects)
plt.ylim(0, 1)
plt.show()


# import seaborn as sns
# from sklearn.metrics import accuracy_score
# def box_plot(y_test, y_pred):
#     # 分类精度数据
#     y_test = np.argmax(y_test, axis=1)
#     # print(y_test,y_pred)
#     acc = accuracy_score(y_test, y_pred)
#     data = [acc]
#
#     # 绘制分布图
#     sns.boxplot(data=data, width=0.5)
#
#     # 设置图表属性
#     plt.title("Classification accuracy distribution")
#     plt.xlabel("MI-EEGNET")
#     plt.ylabel("Accuracy")
#     plt.show()