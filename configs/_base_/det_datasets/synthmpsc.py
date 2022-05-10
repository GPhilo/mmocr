dataset_type = 'IcdarDataset'
data_root = 'data/synthmpsc'

train = dict(
    type=dataset_type,
    ann_file=f'{data_root}/instances_all.json',
    img_prefix=f'{data_root}/gen_pictures',
    pipeline=None)

train_list = [train]
