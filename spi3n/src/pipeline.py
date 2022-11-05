import numpy as np
import torch


def make_loaders(n_img, d_shares, bs, DataSetClass, dataset_params, n_workers, random_state=12345):
    """
    :n_img: number of images total
    :d_shares: shares for train/valid/train split
    :bs: batch size for dataloaders
    :DataSetClass: object with dataset class
    :dataset_params: additional parameters for DataSetClass
    :n_workers: number of cpu for async multiproccess data loading
    :random_state: random_state
    --
    :return: tuple of dataloaders
    (train, valid, test)
    
    """
    
    np.random.seed(random_state)
    images = np.arange(n_img)
    images_perm = np.random.permutation(images)
    
    pos = (np.cumsum(d_shares)/np.sum(d_shares) * n_img).astype(int)
    img_arr_train = images_perm[:pos[0]]
    img_arr_valid = images_perm[pos[0]:pos[1]]
    img_arr_test = images_perm[pos[1]:]

    train_dataset = DataSetClass(
        img_arr = img_arr_train,
        **dataset_params
    )
    val_dataset = DataSetClass(
        img_arr = img_arr_valid,
        **dataset_params
    )
    test_dataset = DataSetClass(
        img_arr = img_arr_test,
        **dataset_params
    )
    
    return \
        torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=n_workers), \
        torch.utils.data.DataLoader(
            val_dataset, batch_size=bs, shuffle=True, num_workers=n_workers), \
        torch.utils.data.DataLoader(
            test_dataset, batch_size=bs, shuffle=False, num_workers=n_workers)
