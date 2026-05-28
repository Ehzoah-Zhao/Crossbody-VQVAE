"""
train_maskgit.py - Train MaskGIT discrete diffusion on VQ token space

Requires: pretrained CrossEmbodimentVQVAE checkpoint
Freezes VQ-VAE, trains MaskGITGenerator on token sequences.
Supports separate generators for robot A and B, or shared generator.
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import torch, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.vqvae as vqvae
import utils.utils_model as utils_model
import options.option_vq as option_vq
from models.maskgit_generator import MaskGITGenerator

from dataset.dataset_A import DATALoader as DataLoaderA
from dataset.dataset_B_v4 import DATALoader as DataLoaderB


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == "__main__":
    # ========== 1. Config ==========
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, args.exp_name + "_maskgit")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)

    import wandb
    wandb.init(project="Crossbody-MaskGIT", name=args.exp_name, config=vars(args))

    # ========== 2. Load & freeze VQ-VAE ==========
    logger.info("Loading VQ-VAE checkpoint: %s", args.ckpt_path)
    vqvae_model = vqvae.CrossEmbodimentVQVAE(
        input_dim_A=134, input_dim_B=363,
        nb_code=args.nb_code, code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t, stride_t=args.stride_t,
        width=args.width, depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.activation, norm=args.norm,
        quantizer_type=args.quantizer_type, mu=args.mu,
    ).cuda()

    ckpt = torch.load(args.ckpt_path, map_location="cuda", weights_only=False)
    vqvae_model.load_state_dict(ckpt.get("net", ckpt), strict=True)
    vqvae_model.eval()
    for p in vqvae_model.parameters():
        p.requires_grad = False
    logger.info("VQ-VAE frozen.")

    # ========== 3. Datasets (for token extraction) ==========
    train_txt = os.path.join(args.split_txt_dir, "train.txt")
    train_loader_A = DataLoaderA(args.data_dir_A, train_txt, args.stat_dir_A, args.batch_size, args.window_size)
    train_loader_B = DataLoaderB(args.data_dir_B, train_txt, args.stat_dir_B, args.batch_size, args.window_size)

    iter_A = cycle(train_loader_A)
    iter_B = cycle(train_loader_B)

    # ========== 4. MaskGIT generator ==========
    # Shared generator for both robots (trained on mixed tokens)
    generator = MaskGITGenerator(
        vocab_size=args.nb_code, code_dim=args.code_dim, max_len=32,
        cond_dim=256, num_layers=6, num_heads=8, dropout=0.1,
    ).cuda()

    gen_optimizer = optim.AdamW(generator.parameters(), lr=args.lr, betas=(0.9, 0.99))
    gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=args.total_iter)

    # ========== 5. Condition embeddings ==========
    cond_A = torch.zeros(args.batch_size, 256, device="cuda")
    cond_B = torch.ones(args.batch_size, 256, device="cuda") * 0.1

    # ========== 6. Training loop ==========
    logger.info("Training MaskGIT generator...")
    for nb_iter in range(args.total_iter):
        # Alternate between A and B tokens
        if nb_iter % 2 == 0:
            motion, _ = next(iter_A)
            motion = motion.cuda().float()
            cond = cond_A[:motion.shape[0]]
            with torch.no_grad():
                tokens = vqvae_model.encode_A(motion)
        else:
            motion, _ = next(iter_B)
            motion = motion.cuda().float()
            cond = cond_B[:motion.shape[0]]
            with torch.no_grad():
                tokens = vqvae_model.encode_B(motion)

        loss = generator.compute_loss(tokens, cond, mask_ratio=0.5)

        gen_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        gen_optimizer.step()
        gen_scheduler.step()

        if nb_iter % args.print_iter == 0:
            gs = nb_iter
            writer.add_scalar("MaskGIT/loss", loss.item(), gs)
            wandb.log({"MaskGIT/loss": loss.item()}, step=gs)
            logger.info("Iter %6d | Loss=%.4f", nb_iter, loss.item())

        if nb_iter % args.save_iter == 0:
            ckpt_path = os.path.join(args.out_dir, "maskgit_iter%07d.pth" % nb_iter)
            torch.save(dict(generator=generator.state_dict(), iter=nb_iter), ckpt_path)
            logger.info("MaskGIT checkpoint -> %s", ckpt_path)

    logger.info("MaskGIT training complete!")