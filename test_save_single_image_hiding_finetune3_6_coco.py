import os
import glob
import random
import argparse
import numpy as np

from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.utils as vutils

import matplotlib.pyplot as plt

from stegformer_crop_model1 import build_crop_robust_model
from critic import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config_finetune3_6_coco')
args_cli = parser.parse_args()

config_module = __import__(args_cli.config)
args = config_module.Args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(args.crop_seed)

IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp']


def get_image_paths(folder):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(folder, ext)))
        paths.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
    return sorted(list(set(paths)))


def load_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert('RGB')
    return img


class PairedImageDataset(Dataset):
    def __init__(self, cover_paths, secret_paths, image_size=256):
        self.cover_paths = cover_paths
        self.secret_paths = secret_paths
        self.image_size = image_size
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.cover_paths)

    def __getitem__(self, index):
        cover_path = self.cover_paths[index]
        secret_path = self.secret_paths[index]
        cover = load_rgb(cover_path)
        secret = load_rgb(secret_path)
        cover = self.to_tensor(cover)
        secret = self.to_tensor(secret)
        cover = F.interpolate(cover.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        secret = F.interpolate(secret.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False).squeeze(0)
        name = os.path.basename(cover_path)
        return cover, secret, name


def build_pairs(test_image_dir=None, cover_dir=None, secret_dir=None, max_images=0, pair_mode='sorted', dataset_mode='single_folder_split'):
    if dataset_mode == 'single_folder_split':
        if test_image_dir is None:
            raise ValueError('dataset_mode=single_folder_split requires test_image_dir.')
        all_paths = get_image_paths(test_image_dir)
        if max_images > 0:
            all_paths = all_paths[:max_images]
        if len(all_paths) < 2:
            raise ValueError('Not enough images in test_image_dir.')
        if len(all_paths) % 2 != 0:
            all_paths = all_paths[:-1]
        half = len(all_paths) // 2
        cover_paths = all_paths[:half]
        secret_paths = all_paths[half:]
        return cover_paths, secret_paths

    if cover_dir is None or secret_dir is None:
        raise ValueError('dataset_mode=paired requires cover_dir and secret_dir.')

    cover_paths = get_image_paths(cover_dir)
    secret_paths = get_image_paths(secret_dir)

    if pair_mode == 'filename':
        secret_dict = {}
        for p in secret_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            secret_dict[stem] = p
        pairs = []
        for cover_path in cover_paths:
            stem = os.path.splitext(os.path.basename(cover_path))[0]
            if stem in secret_dict:
                pairs.append((cover_path, secret_dict[stem]))
        if len(pairs) == 0:
            pair_mode = 'sorted'
        else:
            cover_paths = [p[0] for p in pairs]
            secret_paths = [p[1] for p in pairs]

    if pair_mode == 'sorted':
        n = min(len(cover_paths), len(secret_paths))
        cover_paths = cover_paths[:n]
        secret_paths = secret_paths[:n]

    if max_images > 0:
        n = min(max_images, len(cover_paths), len(secret_paths))
        cover_paths = cover_paths[:n]
        secret_paths = secret_paths[:n]

    return cover_paths, secret_paths


def simulate_crop(x, ratio, seed, fill_mode='mean'):
    if ratio <= 0:
        valid = torch.ones_like(x)
        return x.clone(), valid

    B, C, H, W = x.shape
    attacked = x.clone()
    valid = torch.ones_like(x)
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)

    for b in range(B):
        target_area = H * W * ratio
        crop_h = max(1, int(np.sqrt(target_area)))
        crop_w = max(1, int(target_area / crop_h))
        crop_h = min(crop_h, H - 1)
        crop_w = min(crop_w, W - 1)
        max_y = H - crop_h
        max_x = W - crop_w
        y = torch.randint(0, max_y + 1, (1,), generator=generator, device=x.device).item() if max_y > 0 else 0
        xx = torch.randint(0, max_x + 1, (1,), generator=generator, device=x.device).item() if max_x > 0 else 0

        if fill_mode == 'zero':
            fill = torch.zeros(C, device=x.device)
        elif fill_mode == 'half':
            fill = torch.ones(C, device=x.device) * 0.5
        elif fill_mode == 'noise':
            fill = torch.rand(C, device=x.device)
        else:
            fill = x[b].mean(dim=(1, 2))

        attacked[b, :, y:y + crop_h, xx:xx + crop_w] = fill.view(C, 1, 1)
        valid[b, :, y:y + crop_h, xx:xx + crop_w] = 0

    return attacked, valid


def tensor_to_np(x):
    x = x.detach().cpu().clamp(0, 1)
    return x[0].permute(1, 2, 0).numpy()


def save_tensor_image(tensor, path):
    tensor = tensor.detach().cpu().clamp(0, 1)
    vutils.save_image(tensor, path)


def calculate_metrics(target, prediction, lpips_model):
    target_cpu = target.detach().cpu().clamp(0, 1)
    prediction_cpu = prediction.detach().cpu().clamp(0, 1)

    psnr = calculate_psnr_skimage(target_cpu, prediction_cpu)
    ssim = calculate_ssim_skimage(target_cpu, prediction_cpu)
    mse = F.mse_loss(prediction_cpu, target_cpu).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(prediction_cpu - target_cpu)).item()

    if lpips_model is not None:
        lpips_device = next(lpips_model.parameters()).device
        target_lpips = target.detach().to(lpips_device).clamp(0, 1)
        pred_lpips = prediction.detach().to(lpips_device).clamp(0, 1)
        target_lpips = target_lpips * 2.0 - 1.0
        pred_lpips = pred_lpips * 2.0 - 1.0
        with torch.no_grad():
            lpips_value = lpips_model(target_lpips, pred_lpips).mean().item()
    else:
        lpips_value = np.nan

    return {
        'psnr': float(np.mean(psnr)),
        'ssim': float(np.mean(ssim)),
        'rmse': float(rmse),
        'mae': float(mae),
        'lpips': float(lpips_value)
    }


def main():
    device = torch.device(args.device)
    print('=' * 80)
    print('StegFormer External Dataset Top-50 Evaluation')
    print('=' * 80)
    print('Device:', device)
    print('Model:', args.model_path)
    print('Test number:', args.test_num)
    print('Top-K:', args.top_k)
    print('Ranking metric:', args.ranking_metric)

    cover_paths, secret_paths = build_pairs(
        test_image_dir=getattr(args, 'test_image_dir', None),
        cover_dir=getattr(args, 'test_cover_dir', None),
        secret_dir=getattr(args, 'test_secret_dir', None),
        max_images=args.max_images,
        pair_mode=args.pair_mode,
        dataset_mode=getattr(args, 'dataset_mode', 'paired')
    )

    dataset = PairedImageDataset(cover_paths, secret_paths, args.image_size_test_single)
    loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print('[Model] Building model...')
    model = build_crop_robust_model(args)
    model = model.to(device)
    model.eval()

    print('[Model] Loading:', args.model_path)
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    print('[Model] Loaded.')

    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
        print('[Metric] LPIPS enabled.')
    except Exception as e:
        print('[Metric] LPIPS unavailable:', e)
        lpips_model = None

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for crop_ratio in args.crop_ratios:
        print('\n')
        print('=' * 80)
        scenario_name = 'no_crop' if crop_ratio == 0 else f'crop_{crop_ratio:.2f}'
        print(f'[{scenario_name}] START')
        print('=' * 80)
        scenario_results = []
        scenario_dir = os.path.join(args.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)

        for image_idx, batch in enumerate(loader):
            cover, secret, names = batch
            cover = cover.to(device, non_blocking=True)
            secret = secret.to(device, non_blocking=True)

            with torch.inference_mode():
                stego, A_enc, _ = model.encode(cover, secret)

            if crop_ratio == 0:
                valid_full = torch.ones_like(stego)
                with torch.inference_mode():
                    dec = model.decode(stego, valid_full, run_deblur=False)
                secret_rec = dec.recovered
                secret_enh = secret_rec
            else:
                recovered_list = []
                enhanced_list = []
                for trial in range(args.num_trials):
                    attacked_stego, valid = simulate_crop(stego, crop_ratio, args.crop_seed + image_idx * 100 + trial, args.invalid_fill)
                    with torch.inference_mode():
                        dec = model.decode(attacked_stego, valid, run_deblur=args.use_deblur)
                    recovered_list.append(dec.recovered)
                    if args.use_deblur and dec.enhanced is not None:
                        enhanced_list.append(dec.enhanced)
                secret_rec = torch.stack(recovered_list, dim=0).mean(dim=0)
                secret_enh = torch.stack(enhanced_list, dim=0).mean(dim=0) if len(enhanced_list) > 0 else None

            cover_metrics = calculate_metrics(cover, stego, lpips_model)
            secret_rec_metrics = calculate_metrics(secret, secret_rec, lpips_model)
            if secret_enh is not None:
                secret_enh_metrics = calculate_metrics(secret, secret_enh, lpips_model)
            else:
                secret_enh_metrics = {'psnr': np.nan, 'ssim': np.nan, 'rmse': np.nan, 'mae': np.nan, 'lpips': np.nan}

            result = {
                'index': image_idx,
                'name': names[0],
                'crop_ratio': crop_ratio,
                'cover_psnr': cover_metrics['psnr'],
                'cover_ssim': cover_metrics['ssim'],
                'cover_rmse': cover_metrics['rmse'],
                'cover_mae': cover_metrics['mae'],
                'cover_lpips': cover_metrics['lpips'],
                'secret_psnr': secret_rec_metrics['psnr'],
                'secret_ssim': secret_rec_metrics['ssim'],
                'secret_rmse': secret_rec_metrics['rmse'],
                'secret_mae': secret_rec_metrics['mae'],
                'secret_lpips': secret_rec_metrics['lpips'],
                'enh_psnr': secret_enh_metrics['psnr'],
                'enh_ssim': secret_enh_metrics['ssim'],
                'enh_rmse': secret_enh_metrics['rmse'],
                'enh_mae': secret_enh_metrics['mae'],
                'enh_lpips': secret_enh_metrics['lpips'],
            }
            scenario_results.append(result)

            if (image_idx + 1) % 50 == 0 or image_idx == 0:
                print(f'[{scenario_name}] {image_idx + 1}/{len(dataset)} CoverPSNR={cover_metrics["psnr"]:.3f} CoverSSIM={cover_metrics["ssim"]:.4f} SecretPSNR={secret_rec_metrics["psnr"]:.3f} SecretSSIM={secret_rec_metrics["ssim"]:.4f} SecretLPIPS={secret_rec_metrics["lpips"]:.4f}')

        all_results[crop_ratio] = scenario_results
        if args.ranking_metric == 'psnr':
            scenario_results_sorted = sorted(scenario_results, key=lambda x: x['secret_psnr'], reverse=True)
        elif args.ranking_metric == 'ssim':
            scenario_results_sorted = sorted(scenario_results, key=lambda x: x['secret_ssim'], reverse=True)
        elif args.ranking_metric == 'lpips':
            scenario_results_sorted = sorted(scenario_results, key=lambda x: x['secret_lpips'])
        else:
            raise ValueError('Unknown ranking metric')

        top_k = min(args.top_k, len(scenario_results_sorted))
        top_results = scenario_results_sorted[:top_k]
        metrics = ['cover_psnr', 'cover_ssim', 'cover_rmse', 'cover_mae', 'cover_lpips', 'secret_psnr', 'secret_ssim', 'secret_rmse', 'secret_mae', 'secret_lpips', 'enh_psnr', 'enh_ssim', 'enh_rmse', 'enh_mae', 'enh_lpips']
        top50_average = {'crop_ratio': crop_ratio, 'num_images': len(scenario_results), 'top_k': top_k}
        for metric in metrics:
            values = np.array([r[metric] for r in top_results if not np.isnan(r[metric])])
            top50_average[metric] = values.mean() if len(values) > 0 else np.nan

        print(f'[{scenario_name}] Top-{top_k} average (Secret -> Recovered):')
        print(f'  PSNR      : {top50_average["secret_psnr"]:.4f}')
        print(f'  SSIM      : {top50_average["secret_ssim"]:.4f}')
        print(f'  RMSE      : {top50_average["secret_rmse"]:.6f}')
        print(f'  MAE       : {top50_average["secret_mae"]:.6f}')
        print(f'  LPIPS     : {top50_average["secret_lpips"]:.6f}')

        import csv
        all_csv = os.path.join(scenario_dir, 'metrics_all.csv')
        with open(all_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=scenario_results[0].keys())
            writer.writeheader()
            writer.writerows(scenario_results)

        top_csv = os.path.join(scenario_dir, f'top_{top_k}.csv')
        with open(top_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=top_results[0].keys())
            writer.writeheader()
            writer.writerows(top_results)

    summary = []
    for crop_ratio in args.crop_ratios:
        results = all_results[crop_ratio]
        top_results = results[:min(args.top_k, len(results))]
        row = {
            'crop_ratio': crop_ratio,
            'num_images': len(results),
            'top_k': len(top_results),
            'Cover_PSNR': np.mean([r['cover_psnr'] for r in top_results]),
            'Cover_SSIM': np.mean([r['cover_ssim'] for r in top_results]),
            'Cover_RMSE': np.mean([r['cover_rmse'] for r in top_results]),
            'Cover_MAE': np.mean([r['cover_mae'] for r in top_results]),
            'Cover_LPIPS': np.mean([r['cover_lpips'] for r in top_results]),
            'Secret_PSNR': np.mean([r['secret_psnr'] for r in top_results]),
            'Secret_SSIM': np.mean([r['secret_ssim'] for r in top_results]),
            'Secret_RMSE': np.mean([r['secret_rmse'] for r in top_results]),
            'Secret_MAE': np.mean([r['secret_mae'] for r in top_results]),
            'Secret_LPIPS': np.mean([r['secret_lpips'] for r in top_results]),
            'Enh_PSNR': np.nanmean([r['enh_psnr'] for r in top_results]),
            'Enh_SSIM': np.nanmean([r['enh_ssim'] for r in top_results]),
            'Enh_RMSE': np.nanmean([r['enh_rmse'] for r in top_results]),
            'Enh_MAE': np.nanmean([r['enh_mae'] for r in top_results]),
            'Enh_LPIPS': np.nanmean([r['enh_lpips'] for r in top_results]),
        }
        summary.append(row)

    import csv
    summary_csv = os.path.join(args.output_dir, 'summary_top50.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print('\n')
    print('=' * 100)
    print('FINAL TOP-50 RESULTS')
    print('=' * 100)
    print(f'{"Crop":>10} {"Cover_P":>10} {"Cover_S":>10} {"Secret_P":>10} {"Secret_S":>10} {"Secret_LPIPS":>12}')
    print('-' * 100)
    for row in summary:
        print(f'{row["crop_ratio"] * 100:>9.0f}% {row["Cover_PSNR"]:>10.4f} {row["Cover_SSIM"]:>10.4f} {row["Secret_PSNR"]:>10.4f} {row["Secret_SSIM"]:>10.4f} {row["Secret_LPIPS"]:>12.6f}')
    print('=' * 100)

    crop_percent = [r['crop_ratio'] * 100 for r in summary]
    psnr_values = [r['Secret_PSNR'] for r in summary]
    plt.figure(figsize=(7, 5))
    plt.plot(crop_percent, psnr_values, marker='o')
    plt.xlabel('Crop Ratio (%)')
    plt.ylabel('PSNR (dB)')
    plt.title('Top-50 Secret Reconstruction PSNR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top50_psnr_curve.png'), dpi=300)
    plt.close()

    ssim_values = [r['Secret_SSIM'] for r in summary]
    plt.figure(figsize=(7, 5))
    plt.plot(crop_percent, ssim_values, marker='o')
    plt.xlabel('Crop Ratio (%)')
    plt.ylabel('SSIM')
    plt.title('Top-50 Secret Reconstruction SSIM')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'top50_ssim_curve.png'), dpi=300)
    plt.close()

    print('[Done] Results saved to:', args.output_dir)
    print('Final CSV:', summary_csv)


if __name__ == '__main__':
    main()
