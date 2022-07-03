partition_strategy = "hetero-dir"

# def record_net_data_stats(y_train, net_dataidx_map):
  
#     net_cls_counts = {}

#     for net_i, dataidx in net_dataidx_map.items():
#         unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
#         tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
#         net_cls_counts[net_i] = tmp
#     # logging.debug('Data statistics: %s' % str(net_cls_counts))
#     fields = [i for i in range(10)]
#     fields.insert(0, 'id')
#     w = csv.DictWriter(open('client_data_distribution.csv', 'w'), fields)
#     for key,val in sorted(net_cls_counts.items()):
#         row = {'id': key}
#         row.update(val)
#         w.writerow(row)
#     return net_cls_counts


def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_emnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    emnist_train_ds = EMNIST_truncated(datadir, train=True, download=True, transform=transform)
    emnist_test_ds = EMNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = emnist_train_ds.data, emnist_train_ds.target
    X_test, y_test = emnist_test_ds.data, emnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'emnist':
        X_train, y_train, X_test, y_test = load_emnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset.lower() == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        # if args.poison_type == "howto":
        #     sampled_indices_train = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365,
        #                                 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389]
        #     sampled_indices_test = [32941, 36005, 40138]
        #     cifar10_whole_range = np.arange(X_train.shape[0])
        #     remaining_indices = [i for i in cifar10_whole_range if i not in sampled_indices_train+sampled_indices_test]
        #     X_train = X_train[sampled_indices_train, :, :, :]
        #     logger.info("@@@ Poisoning type: {} Num of Remaining Data Points (excluding poisoned data points): {}".format(
        #                                 args.poison_type, 
        #                                 X_train.shape[0]))
        
        # # 0-49999 normal cifar10, 50000 - 50735 wow airline
        # if args.poison_type == 'southwest+wow':
        #     with open('./saved_datasets/wow_images_new_whole.pkl', 'rb') as train_f:
        #         saved_wow_dataset_whole = pickle.load(train_f)
        #     X_train = np.append(X_train, saved_wow_dataset_whole, axis=0)
        n_train = X_train.shape[0]

    elif dataset == 'cinic10':
        _train_dir = './data/cinic10/cinic-10-trainlarge/train'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        trainset = ImageFolderTruncated(_train_dir, transform=transforms.Compose([transforms.ToTensor(),
                                                            transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), 
                                                                                            requires_grad=False),
                                                                                            (4,4,4,4),mode='reflect').data.squeeze()),
                                                            transforms.ToPILImage(),
                                                            transforms.RandomCrop(32),
                                                            transforms.RandomHorizontalFlip(),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=cinic_mean,std=cinic_std),
                                                            ]))
        y_train = trainset.get_train_labels
        n_train = y_train.shape[0]
    elif dataset == "shakespeare":
        net_dataidx_map = {}
        with open(datadir[0]) as json_file:
            train_data = json.load(json_file)

        with open(datadir[1]) as json_file:
            test_data = json.load(json_file)

        for j in range(n_nets):
            client_user_name = train_data["users"][j]

            client_train_data = train_data["user_data"][client_user_name]['x']
            num_samples_train = len(client_train_data)
            net_dataidx_map[j] = [i for i in range(num_samples_train)] # TODO: this is a dirty hack. needs modification
        return None, net_dataidx_map, None

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while (min_size < 10) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        if dataset == 'cifar10':
            if args.poison_type == 'howto' or args.poison_type == 'greencar-neo':
                green_car_indices = [874, 49163, 34287, 21422, 48003, 47001, 48030, 22984, 37533, 41336, 3678, 37365, 19165, 34385, 41861, 39824, 561, 49588, 4528, 3378, 38658, 38735, 19500,  9744, 47026, 1605, 389] + [32941, 36005, 40138]
                #sanity_check_counter = 0
                for k, v in net_dataidx_map.items():
                    remaining_indices = [i for i in v if i not in green_car_indices]
                    #sanity_check_counter += len(remaining_indices)
                    net_dataidx_map[k] = remaining_indices

            #logger.info("Remaining total number of data points : {}".format(sanity_check_counter))
            # sanity check:
            #aggregated_val = []
            #for val in net_dataidx_map.values():
            #    aggregated_val+= val
            #black_box_indices = [i for i in range(50000) if i not in aggregated_val]
            #logger.info("$$$$$$$$$$$$$$ recovered black box indices: {}".format(black_box_indices))
            #exit()
    # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return net_dataidx_map

num_nets = 10
net_dataidx_map = partition_data(
        "emnist", './data', partition_strategy,
        num_nets, 0.5, args)