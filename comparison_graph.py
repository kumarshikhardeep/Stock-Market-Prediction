import numpy as np
import matplotlib.pyplot as plt
from stock_LSTM import history
from stock_MLP import history2
from stock_LSTM_CONV1D import history3
# data to plot
n_groups = 3
train_acc = [history.history['acc'][-1],history2.history['acc'][-1],history3.history['acc'][-1]]
test_acc = [history.history['val_acc'][-1],history2.history['val_acc'][-1],history3.history['val_acc'][-1]]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
 
rects1 = plt.bar(index, train_acc, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train')
 
rects2 = plt.bar(index + bar_width, test_acc, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test')
 
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Comparison of 3 models')
plt.xticks(index + bar_width, ('LSTM', 'MLP', 'LSTM-CNN'))
plt.legend()
 
plt.tight_layout()
plt.show()
