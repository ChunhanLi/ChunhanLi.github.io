### GAUC
### https://blog.csdn.net/hnu2012/article/details/87892368

### quick_one
def gauc(labels, preds, uids):
    """Calculate group auc
    :param labels: list
    :param predict: list
    :param uids: list
    >>> gauc([1,1,0,0,1], [0, 0,1,0,1], ['a', 'a','a', 'b', 'b'])
    0.4
    >>> gauc([1,1,0,0,1], [1,1,0,0,1], ['a', 'a','a', 'b', 'b'])
    1.0
    >>> gauc([1,1,1,0,0], [1,1,0,0,1], ['a', 'a','a', 'b', 'b'])
    0.0
    >>> gauc([1,1,1,0,1], [1,1,0,0,1], ['a', 'a','a', 'b', 'b'])
    1.0
    """
    assert len(uids) == len(labels)
    assert len(uids) == len(preds)
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        uid = uids[idx]
        group_score[uid].append(preds[idx])
        group_truth[uid].append(truth)

    total_auc = 0
    impression_total = 0
    for user_id in group_truth:
        if label_with_xor(group_truth[user_id]):
            auc = roc_auc_score(np.asarray(
                group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = (float(total_auc) /
                 impression_total) if impression_total else 0
    group_auc = round(group_auc, 6)
    return group_auc


def label_with_xor(lists):
    """
    >>> label_with_xor([1,1,1])
    False
    >>> label_with_xor([0,0,0])
    False
    >>> label_with_xor([0,])
    False
    >>> label_with_xor([1,])
    False
    >>> label_with_xor([0,1])
    True
    """
    if not lists:
        return False
    first = lists[0]
    for i in range(1, len(lists)):
        if lists[i] != first:
            return True
    return False


### a little slow
### ------------->https://github.com/MgtvAi/CompetitionRcDemo/blob/master/base_train.py
def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc

