import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.utils import *
from DPT.dpt.models import DPTDepthModel
import torchvision.transforms as T
from scipy.ndimage import median_filter
import DPT.util.io
# BLIP
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    # results/workspace
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('--test', action='store_true', help="test mode")
    # parser.add_argument('--final', action='store_true', help="final train mode")
    parser.add_argument('--final', type=int, default=10,  help="final train mode")
    parser.add_argument('--refine', action='store_true', help="refine mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=10, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--seed', type=int, default=0)
    # parser.add_argument('--depth_model', type=str, default='dpt_hybrid', help='choose from [dpt_large, dpt_hybrid]')
    parser.add_argument('--guidance_scale', type=float, default=10)
    parser.add_argument('--need_back', action='store_true', help="use back text prompt")
    parser.add_argument('--suppress_face', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--ref_path', default='/root/autodl-tmp/Make-It-3D-master/demo/yizi.png', type=str, help="use image as referance, only support alpha image")

    ### training options
    parser.add_argument('--iters', type=int, default=5000, help="training iters")
    parser.add_argument('--refine_iters', type=int, default=3000, help="refine iters")
    parser.add_argument('--lr', type=float, default=1e-3, help="max learning rate")
    parser.add_argument('--min_lr', type=float, default=1e-4, help="minimal learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=512, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=32, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--albedo_iters', type=int, default=1000, help="training iters that only use albedo shading")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5, help="likelihood of sampling camera location uniformly on the sphere surface area")
    parser.add_argument('--diff_iters', type=int, default=400, help="training iters that only use albedo shading")
    parser.add_argument('--step_range', type=float, nargs='*', default=[0.2, 0.6])
    
    # model options
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--blob_density', type=float, default=5, help="max (center) density for the gaussian density blob")
    parser.add_argument('--blob_radius', type=float, default=0.1, help="control the radius for the gaussian density blob")
    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--backbone', type=str, default='tcnn', choices=['grid', 'tcnn', 'sdf', 'vanilla', 'normal'], help="nerf backbone")
    parser.add_argument('--optim', type=str, default='adan', choices=['adan', 'adam', 'adamw'], help="optimizer")
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=128, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=128, help="render height for NeRF in training")
    
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.1, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.0, 1.5], help="training camera radius range")
    parser.add_argument('--fov', type=float, default=20, help="training camera fovy range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[15, 25], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[70, 110], help="training camera phi range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[0, 360], help="training camera phi range")
    
    parser.add_argument('--lambda_entropy', type=float, default=1, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=1e-3, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=1e-2, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=1, help="loss scale for surface smoothness")
    parser.add_argument('--lambda_img', type=float, default=1e3, help="loss scale for ref loss")
    parser.add_argument('--lambda_depth', type=float, default=1, help="loss scale for depth loss")
    parser.add_argument('--lambda_clip', type=float, default=1, help="loss scale for clip loss")
    
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--light_theta', type=float, default=60, help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")
    parser.add_argument('--max_depth', type=float, default=10.0, help="farthest depth")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     ori_img_path = "/root/autodl-tmp/Make-It-3D-master/demo/true_mogu_yuyi_512.png"
#     ori_img = torch.tensor(cv2.imread(ori_img_path)).float().permute(2, 0, 1).unsqueeze(0).to(device)

#     eeg_latent_path = "/root/autodl-tmp/Make-It-3D-master/tensor_file.pt"
#     eeg_latent = torch.load(eeg_latent_path)

#     # # 创建原始张量
#     # original_tensor = torch.randn(5, 77, 767)

#     # 计算填充尺寸
#     # 假设我们希望扩展到稍大的尺寸
#     target_height = 512
#     target_width = 512
#     channels = 3

#     # 第一步：选择填充尺寸
#     # 我们需要将77填充到最接近512的尺寸，767填充到最接近512的尺寸
#     # 选择填充前后尽可能平衡
#     padding_left = (target_width - 767) // 2
#     padding_right = target_width - 767 - padding_left
#     padding_top = (target_height - 77) // 2
#     padding_bottom = target_height - 77 - padding_top

#     # 使用torch.nn.functional.pad进行填充
#     padded_tensor = F.pad(eeg_latent, (padding_left, padding_right, padding_top, padding_bottom))

#     # 第二步：调整维度以匹配目标尺寸
#     # 我们有5个这样的填充后的图，需要将它们转换成3个通道的图像
#     # 这里使用mean减少维度，这是一个简化的示例，实际操作可能需要更复杂的处理
#     resized_tensor = padded_tensor.mean(dim=0, keepdim=True).repeat(3, 1, 1).unsqueeze(0).to(device)

#     print(resized_tensor.shape)  # 输出应该是 torch.Size([1, 3, 512, 512])




#     clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, jit=False)
#     #
#     aug = T.Compose([
#         T.Resize((224, 224)),
#         T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#     ])

#     #
#     image_z_1 = clip_model.encode_image(aug(ori_img))
#     image_z_2 = clip_model.encode_image(aug(resized_tensor))

#     loss = 1 - torch.cosine_similarity(image_z_1, image_z_2, dim=-1).mean()


    opt = parser.parse_args()
    opt.cuda_ray = True
    optDict = opt.__dict__
    opt.workspace = os.path.join('results', opt.workspace)
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True) 
    
    if opt.backbone == 'vanilla':
        from nerf.network import NeRFNetwork
    elif opt.backbone == 'tcnn':
        from nerf.network_tcnn import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    print(opt)
    seed_everything(opt.seed)

    # load depth network
    net_w = net_h = 384
    depth_model = DPTDepthModel(
        path="dpt_weights/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_transform = T.Compose(
    [
        T.Resize((384, 384)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    )


    depth_model.to(device)

    if opt.optim == 'adan':
        from optimizer import Adan
        # Adan usually requires a larger LR
        optimizer = lambda model: Adan(model.get_params(5 * opt.lr), eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    else: # adam
        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

    if opt.backbone == 'vanilla':
        warm_up_with_cosine_lr = lambda iter: iter / opt.warm_iters if iter <= opt.warm_iters \
            else max(0.5 * ( math.cos((iter - opt.warm_iters) /(opt.iters - opt.warm_iters) * math.pi) + 1), 
                        opt.min_lr / opt.lr)

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_cosine_lr)
    else:
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1) # fixed
        # scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    if opt.guidance == 'stable-diffusion':
        from nerf.sd import StableDiffusion
        guidance = StableDiffusion(device, opt.sd_version, opt.hf_key, step_range=opt.step_range)
    elif opt.guidance == 'clip':
        from nerf.clip import CLIP
        guidance = CLIP(device)
    else:
        raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

    ref_imgs = cv2.imread(opt.ref_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
    image_pil = Image.open(opt.ref_path).convert("RGB")

    # generated caption
    if opt.text == None:
        # opt.text = "Caption:  a brown teddy bear sitting on a ground"
        # opt.text = "Caption:  a train traveling on the tracks with cargo"
        # opt.text = " Caption:  a white mustang convertible car on a ground"  
        # opt.text =  "Caption:  a mushroom is shown on a ground"
        # opt.text =  "Caption:  a silver convertible car on a ground"
        # opt.text =  "Caption:  a white volkswagen beetle convertible with the top down"
        # opt.text =   "Caption:  a red corvette convertible car on a ground"
        # opt.text =   "Caption:  a large blue airplane flying in the sky"
        # opt.text =   "a large blue and white airplane on a blue background"
        # opt.text =   "Caption:  a large jet airplane flying in the air"
        print("load blip2 for image caption...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")
        inputs = processor(image_pil, return_tensors="pt").to("cuda", torch.float16)
        out = blip_model.generate(**inputs)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        caption = caption.replace("there is ", "")
        caption = caption.replace("close up", "photo")
        for d in ["black background", "white background"]:
            if d in caption:
                caption = caption.replace(d, "ground")
        print("Caption: ", caption)
        opt.text = caption

    with open(os.path.join(opt.workspace, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in optDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


    # only support alpha photo input.
    imgs = cv2.cvtColor(ref_imgs, cv2.COLOR_BGRA2RGBA)
    imgs = cv2.resize(imgs, (512, 512), interpolation=cv2.INTER_AREA)
    ref_imgs = (torch.from_numpy(imgs)/255.).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    ori_imgs = ref_imgs[:, :3, :, :] * ref_imgs[:, 3:, :, :] + (1 - ref_imgs[:, 3:, :, :])
    
    mask = imgs[:, :, 3:]
    # mask[mask < 0.5 * 255] = 0
    # mask[mask >= 0.5 * 255] = 1 
    kernel = np.ones(((5,5)), np.uint8) ##11
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = (mask == 0)
    mask = (torch.from_numpy(mask)).unsqueeze(0).unsqueeze(0).to(device)
    depth_mask = mask
    
    # depth estimation
    with torch.no_grad():
        depth_prediction = depth_model.forward(depth_transform(ori_imgs))
        depth_prediction = torch.nn.functional.interpolate(
            depth_prediction.unsqueeze(1),
            size=512,
            mode="bicubic",
            align_corners=True,
        ) # [1, 1, 512, 512] [80~150]
        DPT.util.io.write_depth_name(os.path.join(opt.workspace, opt.text.replace(" ", "_") + '_depth'), depth_prediction.squeeze().cpu().numpy(), bits=2)
        disparity = imageio.imread(os.path.join(opt.workspace, opt.text.replace(" ", "_") + '_depth.png')) / 65535.
        disparity = median_filter(disparity, size=5)
        depth = 1. / np.maximum(disparity, 1e-2)

    depth_prediction = torch.tensor(depth, device=device)
    depth_mask = torch.tensor(depth_mask, device=device)
    # normalize estimated depth
    depth_prediction = depth_prediction * (~depth_mask) + torch.ones_like(depth_prediction) * (depth_mask)
    depth_prediction = ((depth_prediction - 1.0) / (depth_prediction.max() - 1.0)) * 0.9 + 0.1
    # save_image(ori_imgs, os.path.join(opt.workspace, opt.text.replace(" ", "_") + '_ref.png'))
    
    
    # torch.save(depth_prediction, "depth_prediction.pt")
    # depth_prediction = torch.load('depth_prediction.pt')


    model = NeRFNetwork(opt)
    trainer = Trainer('df', opt, model, depth_model, guidance, 
                        ref_imgs=ref_imgs, ref_depth=depth_prediction, 
                        ref_mask=depth_mask, ori_imgs=ori_imgs, 
                        device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)
    

    if opt.test:
        test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=33).dataloader()
        trainer.test(test_loader, write_video=True)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)

    else:

        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, size=100).dataloader()
        valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, size=5).dataloader()
        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        if opt.final:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=100).dataloader()
            trainer.test(test_loader, write_image=True, write_video=True)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)

        if opt.refine:
            mv_loader = NeRFDataset(opt, device=device, type='gen_mv', H=opt.H, W=opt.W, size=33).dataloader()
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, size=64).dataloader()
            trainer.test(mv_loader, save_path=os.path.join(opt.workspace, 'mvimg'), write_image=True, write_video=False)
            trainer.refine(os.path.join(opt.workspace, 'mvimg'), opt.refine_iters, test_loader)
        