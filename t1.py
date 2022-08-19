import numpy as np

# s1 = np.load("./stylegan_bedroom3/style00.npy", allow_pickle=True)
# s2 = np.load("./stylegan_bedroom3/style01.npy", allow_pickle=True)
# a = np.load("./stylegan_bedroom3/attribute.npy", allow_pickle=True)
# print(a.dtype)
# print(a)
# print(s1[0].size)
# print(s2[0].size)


# scores = np.load("./stylegan_bedroom3/attribute.npy", allow_pickle=True)[()]
# score_name = 'stressful'
# print(isinstance(scores, dict))
# print(scores)
# print(scores['score'].size)
# if "indoor_lighting" in scores:
#     print('1')
#     scores = scores['indoor_lighting']
#     print(scores)
# else:
#     print('2')
#     score_idx = scores['name_to_idx'][score_name]
#     scores = scores['score'][:, score_idx]
#     print(score_idx)
#     print(scores)
#     print(scores.size)

# attribute = np.load("./attribute.npy", allow_pickle=True)[()]
# print(attribute)
# print(attribute['score'][0].size)
#
# boundary = np.load("./indoor_lighting_w_boundary.npy")
# print(boundary[0].size)
#
# w = np.load("./w.npy")
# print(w[])
from sklearn import svm

data = np.load("./b/w.npy")
print(data.size)
scores = np.load("./b/attribute.npy", allow_pickle=True)[()]
if "indoor_lighting" in scores:
    scores = scores["indoor_lighting"]
else:
    score_idx = scores["name_to_idx"]["indoor_lighting"]
    print(score_idx)
    scores = scores['score'][:, score_idx]
data_shape = data.shape
print(data_shape)
boundaries = []
num, dim = data.shape
#scores = scores[:, 0]
_sorted_idx = np.argsort(scores)[::-1]
print(_sorted_idx)
data = data[_sorted_idx]
print(data)
scores = scores[_sorted_idx]
print(scores)
num = scores.shape[0]

chosen_num=2000
chosen_num=min(chosen_num, num//2)
print(chosen_num)
positive_idx = np.arange(0,chosen_num)
negative_idx = np.arange(num-chosen_num, num)
remaining_idx = np.arange(chosen_num, num-chosen_num)
print(positive_idx)
print(negative_idx)
print(remaining_idx)
positive_num = positive_idx.size
negative_num = negative_idx.size
remaining_num = num - positive_num - negative_num
positive_train_num = int(positive_num * 0.7)
negative_train_num = int(negative_num * 0.7)
train_num = positive_train_num + negative_train_num
print(train_num)
positive_val_num = positive_num - positive_train_num
negative_val_num = negative_num - negative_train_num
val_num = positive_val_num + negative_val_num
print(val_num)
np.random.shuffle(positive_idx)
np.random.shuffle(negative_idx)
positive_train_idx = positive_idx[:positive_train_num]
negative_train_idx = negative_idx[:negative_train_num]
positive_val_idx = positive_idx[positive_train_num:]
negative_val_idx = negative_idx[negative_train_num:]
train_idx = np.concatenate([positive_train_idx, negative_train_idx])
val_idx = np.concatenate([positive_val_idx, negative_val_idx])
train_data = data[train_idx]
val_data = data[val_idx]

train_scores = scores[train_idx]
val_scores = scores[val_idx]

separation_idx = positive_train_num
is_score_ascending=False

num = train_scores.shape[0]
print(num)
labels = np.zeros(num, dtype=np.bool)
print(labels)
print(separation_idx)
if separation_idx is not None:
    separation_idx = np.clip(separation_idx, 0, num)
    print(separation_idx)
    if is_score_ascending:
        labels[separation_idx:] = True
    else:
        labels[:separation_idx] = True
train_labels = labels
print(train_labels.size)

separation_idx = positive_val_num
is_score_ascending=False

num = val_scores.shape[0]
print(num)
labels = np.zeros(num, dtype=np.bool)
if separation_idx is not None:
    separation_idx = np.clip(separation_idx, 0, num)
    print(separation_idx)
    if is_score_ascending:
        labels[separation_idx:] = True
    else:
        labels[:separation_idx] = True
val_labels = labels
print(val_labels.size)
print(train_num)
print(positive_train_num)
print(negative_train_num)
print(val_num)
print(positive_val_num)
print(negative_val_num)

clf = svm.SVC(kernel='linear')
classifier = clf.fit(train_data, train_labels)
direction = classifier.coef_.reshape(1, dim).astype(np.float32)
print(classifier.coef_.shape)
boundary = direction / np.linalg.norm(direction)
print(boundary.shape)



# attribute = np.load("./a/indoor_lighting_boundary.npy", allow_pickle=True)[()]
# print(attribute)