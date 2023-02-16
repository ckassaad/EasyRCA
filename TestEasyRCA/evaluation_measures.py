def tp(lhat, ltrue):
    ltp = [i for i in lhat if i in ltrue]
    return len(ltp)


def fp(lhat, ltrue):
    lfp = [i for i in lhat if i not in ltrue]
    return len(lfp)


def fn(lhat, ltrue):
    lfn = [i for i in ltrue if i not in lhat]
    return len(lfn)


def precision(lhat, ltrue):
    true_pos = tp(lhat, ltrue)
    false_pos = fp(lhat, ltrue)
    if true_pos == 0:
        return 0
    else:
        return true_pos / (true_pos + false_pos)


def recall(lhat, ltrue):
    true_pos = tp(lhat, ltrue)
    false_neg = fn(lhat, ltrue)
    if true_pos == 0:
        return 0
    else:
        return true_pos / (true_pos + false_neg)


def f1(lhat, ltrue):
    p = precision(lhat, ltrue)
    r = recall(lhat, ltrue)
    if (p == 0) and (r == 0):
        return 0
    else:
        return 2 * p * r / (p + r)
