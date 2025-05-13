import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
np.random.seed(0)
ids = np.arange(1, 21)
true_label = np.random.randint(0, 2, 20)
pred_model1 = true_label + np.random.normal(0, 0.1, 20)
pred_model2 = true_label + np.random.normal(0, 0.2, 20)

# 绘制折线图
plt.plot(ids, true_label, marker='o', color='blue', label='True Label')
plt.plot(ids, pred_model1, marker='x', color='red', label='Model 1')
plt.plot(ids, pred_model2, marker='^', color='green', label='Model 2')

plt.xlabel('ID')
plt.ylabel('Predicted Value')
plt.legend()

plt.savefig('model_comparison_distance.png')
