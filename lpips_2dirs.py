import argparse
import os
import lpips
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1', '--dir1', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o', '--out', type=str, default='./imgs/example_dists.txt')
parser.add_argument('-v', '--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# Initialize the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)
device = torch.device("cuda" if opt.use_gpu and torch.cuda.is_available() else "cpu")
loss_fn.to(device)

# Open output file
with open(opt.out, 'w') as f:
    files = os.listdir(opt.dir0)
    csim_sum = 0
    count = 0

    for file in files:
        img0_path = os.path.join(opt.dir0, file)
        img1_path = os.path.join(opt.dir1, file)

        if os.path.exists(img1_path):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(img0_path)).to(device)
            img1 = lpips.im2tensor(lpips.load_image(img1_path)).to(device)

            # Resize images to (512, 512)
            img0 = F.interpolate(img0, size=(512, 512), mode='bilinear', align_corners=False)
            img1 = F.interpolate(img1, size=(512, 512), mode='bilinear', align_corners=False)

            # Compute LPIPS distance
            with torch.no_grad():  
                dist01 = loss_fn(img0, img1).item()  

            csim_sum += dist01
            count += 1
            print(f"{file}: {dist01:.3f} (Processed: {count})")

            f.write(f"{file}: {dist01:.6f}\n")

            # remove useless image
            del img0, img1, dist01
            torch.cuda.empty_cache()  # clean cache

    # calculte csim
    avg_csim = csim_sum / count if count > 0 else 0
    print(f"Work completed!! The avg csim = {avg_csim:.6f}!!")
