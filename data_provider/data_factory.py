from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader


def data_provider(args, flag):
    data_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=0 if args.embed != 'timeF' else 1,
        freq=args.freq,
        percent=args.percent,
        seasonal_patterns=args.seasonal_patterns,
        train_rate=args.train_rate
    )
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False if flag == 'test' else True,
        num_workers=args.num_workers,
        drop_last=False if flag == 'test' else True
    )
    return data_set, data_loader
