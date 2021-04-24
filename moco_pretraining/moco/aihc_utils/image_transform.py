import torchvision.transforms as transforms

CXR_MEAN = [.5020, .5020, .5020]
CXR_STD = [.085585, .085585, .085585]


def get_transform(args, training):
    # Shorter side scaled to args.img_size
    if args.maintain_ratio:
        transforms_list = [transforms.Resize(args.img_size)]
    else:
        transforms_list = [transforms.Resize((args.img_size, args.img_size))]

    # Data augmentation
    if training:
        transforms_list += [transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(args.rotate), 
                            transforms.RandomCrop((args.crop, args.crop)) if args.crop != 0 else None]
    else:
        transforms_list += [transforms.CenterCrop((args.crop, args.crop)) if args.crop else None]

    # Normalization
    # Seems like the arguments do not contain clahe anyways
    # if t_args.clahe:
    #     transforms_list += [CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))]

    normalize = transforms.Normalize(mean=CXR_MEAN, std=CXR_STD)
    transforms_list += [transforms.ToTensor(), normalize]

    # transform = transforms.Compose([t for t in transforms_list if t])
    transform = [t for t in transforms_list if t]
    return transform