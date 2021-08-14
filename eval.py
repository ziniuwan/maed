import os
import torch
import torchvision

from lib.dataset import VideoDataset
from lib.data_utils.transforms import *
from lib.models import MAED
from lib.core.evaluate import Evaluator
from lib.core.config import parse_args
from torch.utils.data import DataLoader

def main(cfg, args):
    print(f'...Evaluating on {args.eval_ds.lower()} {args.eval_set.lower()} set...')
    device = "cuda"

    model = MAED(
        encoder=cfg.MODEL.ENCODER.BACKBONE, 
        num_blocks=cfg.MODEL.ENCODER.NUM_BLOCKS, 
        num_heads=cfg.MODEL.ENCODER.NUM_HEADS, 
        st_mode=cfg.MODEL.ENCODER.SPA_TEMP_MODE,
        decoder=cfg.MODEL.DECODER.BACKBONE, 
        hidden_dim=cfg.MODEL.DECODER.HIDDEN_DIM,
    )
    model = model.to(device)

    if args.pretrained != '' and os.path.isfile(args.pretrained):
        checkpoint = torch.load(args.pretrained)
        best_performance = checkpoint['performance']
        checkpoint['state_dict'] = {k[len('module.'):]: w for k, w in checkpoint['state_dict'].items() if k.startswith('module.') and not 'smpl' in k}
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f'==> Loaded pretrained model from {args.pretrained}...')
        print(f'==> Performance on 3DPW test set {best_performance}')
    else:
        print(f'{args.pretrained} is not a pretrained model!!!!')
        exit()

    transforms = torchvision.transforms.Compose([
        CropVideo(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH),
        StackFrames(),
        ToTensorVideo(),
        NormalizeVideo(),
    ])

    test_db = VideoDataset(
        args.eval_ds.lower(),
        set=args.eval_set.lower(),
        transforms=transforms,
        sample_pool=cfg.EVAL.SAMPLE_POOL,
        random_sample=False, random_start=False,
        verbose=True,
        debug=cfg.DEBUG)

    test_loader = DataLoader(
        dataset=test_db,
        batch_size=cfg.EVAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    Evaluator().run(
        model=model,
        dataloader=test_loader,
        seqlen=cfg.EVAL.SEQLEN,
        interp=cfg.EVAL.INTERPOLATION,
        save_path=args.output_path,
        device=cfg.DEVICE,
    )


if __name__ == '__main__':
    args, cfg, cfg_file = parse_args()

    main(cfg, args)

