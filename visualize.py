import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

n_files = [os.path.join('logging/eps10', i) for i in os.listdir('logging/eps10')]

meta_r1 = pd.read_csv('logging/new_benchmark_01.csv', names=["flr", "isAttacker", "id", "bias"])
ids = meta_r1['id'].unique()
meta_r1 = meta_r1.set_index('id')
print(meta_r1)
dicts = {}
diffs = {}

for id in ids:
  name = f'{id}_net.pth'
  path = os.path.join('logging/eps10', name)
  dicts[id] = torch.load(path)

avg_prev = dicts[-2]

for k in dicts.keys():
  d = dicts[k]
  dif_w = d['fc3.weight'] - avg_prev['fc3.weight']
  dif_b = d['fc3.bias'] - avg_prev['fc3.bias']
  diff = torch.cat((dif_w.flatten(),dif_b))
  diffs[k] =  diff.detach().cpu().numpy()

# weights = []
# for d in dicts:
#   dif_w = d['fc3.weight'] - avg_prev['fc3.weight']
#   dif_b = d['fc3.bias'] - avg_prev['fc3.bias']
#   diff = torch.cat((dif_w.flatten(),dif_b))

#   diffs.append(diff.detach().cpu().numpy())

# print(diff.shape)
# start = -1290
# end = -1
# print(diffs)

for i, diff in diffs.items():
  # if meta_r1.loc[i]['id'] in [-2, -1]:
  #   continue
  # if i in [-2, -1]:
  #   continue
  # if meta_r1.loc[i]['isAttacker'] == 0:
  #   continue
  plt.figure(figsize=(20,4))
  plt.bar(x=range(1290), height=diff)
  plt.xlabel('Param index')
  plt.ylabel('Shifted value')

  v_x = [128 * i for i in range(11)]
  plt.vlines(v_x, ymin=-0.25, ymax=0.25, linestyles='dashed', color='red', linewidth=0.5)
  plt.ylim((-0.25,0.25))
  plt.title(f"id: {i}, is attacker: {'false' if meta_r1.loc[i]['isAttacker'] == 0 else 'true'}")
#   plt.show()
  plt.savefig(f'logging/eps10/{i}_net.png')
