from argparse import ArgumentParser
from module import LayoutSegmentation
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import CornerDataset

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=-1, help='Number of GPUs')
    parser.add_argument('--ckpt', required=True, type=str, help="Path to checkpoint.")
    parser.add_argument('--dataset_path', required=True, type=str, help="Path to datasets.")

    args = parser.parse_args()

    """
    src = cv2.imread(args.img)
    if src is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    # Convert image to gray and blur it
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    src_edge = edges(src_gray)
    corners = find_corner(src_edge)

    src_corner = cv2.cvtColor(src_edge, cv2.COLOR_GRAY2BGR)

    for c in corners:
        cv2.circle(src_corner, c, 5, (0,255,0), -1)
    

    cv2.imshow("img", src)
    cv2.imshow("src_corner", src_corner)
    cv2.waitKey(0)
    """

    args = parser.parse_args()

    use_gpu = not args.gpus == 0

    trainer = pl.Trainer(
        gpus=args.gpus,
        precision=args.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None)

    model = LayoutSegmentation.load_from_checkpoint(args.ckpt)

    test_dataset   = CornerDataset(path=args.dataset_path, split="test", scale=0.5)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    trainer.test(model=model, test_dataloaders=test_loader)