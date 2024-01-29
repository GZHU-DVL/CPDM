"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
th.set_num_threads(1)

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location=dist_util.dev())
    )
    model.to(dist_util.dev())
    print(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,  # deterministic if True, yield results in a deterministic order.
    )
    data_gt = load_data(
        data_dir=args.data_gt_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,  # deterministic if True, yield results in a deterministic order.
    )
    logger.log("sampling...")
    all_images = []
    all_labels = []
    flag = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        gt, _ = next(data_gt)
        gt = ((gt + 1) * 127.5).clamp(0, 255).to(th.uint8)
        gt = gt.permute(0, 2, 3, 1)
        g = gt.cpu().detach().numpy()
        for i in range(args.batch_size):
            res = g[i]
            image = Image.fromarray(np.uint8(res)).convert('RGB')
            # image.show()
            # timestamp = datetime.datetime.now().strftime("%M-%S")
            savepath = r'./eval64/gt/{}_{}gt.png'.format(
                flag, i)
            image.save(savepath)
        input_data, _ = next(data)
        style = ((input_data + 1) * 127.5).clamp(0, 255).to(th.uint8)
        style = style.permute(0, 2, 3, 1)
        s = style.cpu().detach().numpy()
        for i in range(args.batch_size):
            res = s[i]
            image = Image.fromarray(np.uint8(res)).convert('RGB')
            # image.show()
            # timestamp = datetime.datetime.now().strftime("%M-%S")
            savepath = r'./eval64/input/{}_{}input.png'.format(
                flag, i)
            image.save(savepath)

        input_data = input_data.cuda()
        print(input_data.device)
        noise = th.rand_like(input_data)

        noise = noise.cuda()
        print(noise.device)    
        noise = th.cat((noise, input_data, input_data-noise), dim=1)
        #noise = th.cat((noise, noise), dim=1)
        
        noise = noise.cuda()
        print(noise.device)
        sample = sample_fn(
            model,
            (args.batch_size, 9, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            noise=noise
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        k = sample.cpu().detach().numpy()
        for i in range(args.batch_size):
            res = k[i]
            image = Image.fromarray(np.uint8(res)).convert('RGB')
            # image.show()
            # timestamp = datetime.datetime.now().strftime("%M-%S")
            savepath = r'./eval64/generate/{}_{}generate.png'.format(
                flag, i)
            image.save(savepath)
        flag += 1
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # if args.class_cond:
        #     gathered_labels = [
        #         th.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    # if args.class_cond:
    #     label_arr = np.concatenate(all_labels, axis=0)
    #     label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
        # if args.class_cond:
        #     np.savez(out_path, arr, label_arr)
        # else:
        #     np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        data_gt_dir="",
        deterministic=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
