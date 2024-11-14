import random
from functools import partial
from itertools import repeat
from typing import Callable
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler

import torch.utils.data
import numpy as np


def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))

def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)        
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_target = next_target.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=False,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=True,
                  worker_seeding='all'):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    return loader