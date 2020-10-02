def rle(mask):
    """ mask is a binary 0/1 or True/False array """
    if not np.any(mask):
        return []
    else:
        mask = mask.reshape(-1, order='F')
        mask = np.concatenate(([False], mask, [False]))
        edges = mask[1:] ^ mask[:-1]
        idxs = np.flatnonzero(edges)  # start-end indices
        idxs += 1  # 1-indexed
        idxs[1::2] = idxs[1::2] - idxs[0::2]  # replace 'end' by 'run'
        return list(idxs)
