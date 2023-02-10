import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms

from models.classifier import Classifier
from models.method import SAM
from tensorboardX import SummaryWriter
from models.resnet import resnet18, resnet34, resnet50, resnet152, resnet101
from src.utils import load_network, load_data, display_instances

#from src.CompactBilinearPooling import CompactBilinearPooling
from compact_bilinear_pooling import CountSketch, CompactBilinearPooling

import skimage
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
# from torchvision.transforms.functional.gaussian_blur import gaussian_blur
import torch.nn.functional as nnf

import cv2

from PIL import Image
from torchvision import models
import argparse
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

import torch.nn.functional as F
import tqdm

torch.autograd.set_detect_anomaly(True)

def get_attentions(vits16, vits_in, x, orig_img=None):

    attentions = vits16.get_last_selfattention(vits_in)

    bs = attentions.shape[0]
    nh = attentions.shape[1] # number of head
    # we keep only the output patch attention
    attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)

    # print("attention sshape : ", attentions.shape)
    # 1/0

    threshold = 0.5
    val, idx = torch.sort(attentions)

    val /= torch.sum(val, dim=2, keepdim=True)

    cumval = torch.cumsum(val, dim=2)

    # print("cum val shape; ", cumval.shape)
    # print("idxL: " , idx[0][0])

        
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)

    # print(idx2.shape)
    # print("idx 2; ", idx2[0][0])
    # print("th attn: ", th_attn.shape)
    # print(th_attn[0][0])
    # 1/0

    for elt in range(bs):
        for head in range(nh):
            th_attn[elt][head] = th_attn[elt][head][idx2[elt][head]]

    # print(th_attn[0][0])
    # 1/0

    w_featmap, h_featmap = 14, 14

    th_attn = th_attn.reshape(bs, nh, w_featmap, h_featmap).float()

    
    th_attn = nn.functional.interpolate(th_attn, scale_factor=16, mode="nearest").detach()
    th_attn = torch.amax(th_attn, dim=1)
    # print("th_attn shape: ", th_attn.shape)
    # 1/0

    # save_image(th_attn[9].cpu(), 'foreground_mask.png')
    # save_image(orig_img[9].cpu(), 'actual_img.png')
    # 1/0

    th_attn = torch.tile(th_attn[:, None, :], (1, 3, 1, 1))
    # print("th_attn shape: ", th_attn.shape)
    # print("x shape; ", x.shape)




    foreground = th_attn * orig_img
    background = (torch.ones(th_attn.shape).cuda() - th_attn) * orig_img

    blurring = GaussianBlur(kernel_size=(11,11), sigma=(10.5, 10.5))
    
    # blurred_img = blurring(orig_img)
    blurred_img = blurring(orig_img)

    other_foreground = torch.where(th_attn > 0, orig_img, blurred_img)

    # save_image(x.cpu(), 'x.png')
    # save_image(orig_img.cpu(), 'orig_img.png')
    # save_image(foreground.cpu(), 'foreground.png')
    # save_image(other_foreground.cpu(), 'other_foreground.png')
    # save_image(background.cpu(), 'background.png')
    # # save_image(orig_img.cpu(), 'actual_img.png')
    # 1/0


    # foreground = th_attn * x
    # background = (torch.ones(th_attn.shape).cuda() - th_attn) * x

    # return th_attn, foreground, background
    return th_attn, other_foreground, background
    1/0

    
    attentions = attentions.reshape(bs, nh, w_featmap, h_featmap)

    # print("attentions shape: ", attentions.shape)

    # we keep only a certain percentage of the mass
    # val, idx = torch.sort(attentions)
    # val /= torch.sum(val, dim=1, keepdim=True)
    # cumval = torch.cumsum(val, dim=1)
    # th_attn = cumval > (1 - args.threshold)
    # idx2 = torch.argsort(idx)
    # for head in range(nh):
    #     th_attn[head] = th_attn[head][idx2[head]]
    # th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
    # # interpolate
    # th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = nn.functional.interpolate(attentions, scale_factor=16, mode="nearest").detach()
    # attentions = nn.functional.interpolate(attentions, scale_factor=0.5, mode="nearest").detach()
    # print("x shape: ", x.shape)

    # plt.imsave(fname="mplt_attention_head_0.png", arr=attentions[0][0].cpu(), format='png')
    # plt.imsave(fname="mplt_attention_head_1.png", arr=attentions[0][1].cpu(), format='png')
    # plt.imsave(fname="mplt_attention_head_2.png", arr=attentions[0][2].cpu(), format='png')
    # plt.imsave(fname="mplt_attention_head_3.png", arr=attentions[0][3].cpu(), format='png')
    # plt.imsave(fname="mplt_attention_head_4.png", arr=attentions[0][4].cpu(), format='png')
    # plt.imsave(fname="mplt_attention_head_5.png", arr=attentions[0][5].cpu(), format='png')

    
    # display_instances(torch.permute(x[0], (1, 2, 0)).cpu().numpy()*255, attentions[0][0].cpu().numpy(), fname="attention_head_0.png", blur=False)

    # print("another shape: ", attentions[0][0].shape, torch.max(attentions[0][0]), torch.min(attentions[0][0]))

    attentions = torch.mean(attentions, dim=1)
    thresh = nn.Threshold(0.005, 0)

    blurring = GaussianBlur(kernel_size=(11,11), sigma=(10.5, 10.5))
    
    # blurred_img = blurring(orig_img)
    blurred_img = blurring(x)

    # attentions = thresh(attentions)
    zero_te = torch.zeros(attentions.shape)
    ones_te = torch.ones(attentions.shape)
    # attentions = torch.where(attentions.cpu() > 0.005, ones_te, zero_te)

    # print("attentions shape: ", attentions.shape)
    # print("x shape: ", x.shape)


    # attentions = torch.where(attentions.cpu() > 0.005, ones_te, zero_te)
    # plt.imsave(fname="mplt_attention_head_avg.png", arr=attentions[0].cpu(), format='png')

    # m = nn.AvgPool2d(2, stride=2)
    # attentions = m(attentions)

    attentions = torch.tile(attentions[:, None, :], (1, 3, 1, 1))

    # other_foreground = torch.where(attentions.cpu() > 0.005, orig_img.cpu(), blurred_img.cpu())
    other_foreground = torch.where(attentions > 0.005, x, blurred_img)
    # other_foreground = torch.where(attentions > 0.005, orig_img, blurred_img)

    other_background = torch.where(attentions <= 0.005, x, blurred_img)
    # other_background = torch.where(attentions < 0.005, orig_img, blurred_img)

    # foreground = attentions*x.cpu()
    foreground = attentions*x
    

    # other_foreground = attentions*orig_img.cpu()
    # save_image(other_foreground[0].cpu(), 'foreground.png')
    # save_image(other_background[0].cpu(), 'background.png')

    # 1/0
    # attentions = torch.reshape(attentions, (attentions.shape[0], -1))
    # print("attentions shape: ", attentions.shape)


    # return attentions, foreground
    return attentions, other_foreground, other_background



def test(loader, model, classifier, device):


    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    vits16 = vits16.to(device)

    imagenet_mean=(0.485, 0.456, 0.406)
    imagenet_std=(0.229, 0.224, 0.225)

    normalize = transforms.Compose([transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    for param in vits16.parameters():
        param.requires_grad = False

    # svit_transform = transforms.Compose([
    #     ResizeImage(crop_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test0'])
        iter_val = [iter(loader['test' + str(i)]) for i in range(10)]
        for _ in range(val_len):
            data = [iter_val[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            # vit_inputs = [data[j][2] for j in range(10)]

            labels = data[0][1]
            for j in range(10):
                inputs[j] = inputs[j].to(device)
            labels = labels.to(device)
            outputs = []

            # print("inputs shape: ", len(inputs), inputs[0].shape)

            

            for j in range(10):
                # vit_inputs = normalize(inputs[j])
                # attentions, foreground, background = get_attentions(vits16, vit_inputs, inputs[j])

                # save_image(inputs[j].cpu(), 'test_input_img.png')

                attentions, foreground, background = get_attentions(vits16, normalize(inputs[j]), normalize(inputs[j]),  inputs[j])
                # save_image(foreground.cpu(), 'test_foreground.png')

                foreground = normalize(foreground)

                # 1/0

                # feat,_ = model.inference(inputs[j]) 

        
                # foreground = normalize(foreground)
                # background = normalize(background)

                feat, _ = model.inference(foreground.to(device))
                # bg_feat, _ = model.inference(background.to(device))

                # feat = torch.concat((feat, attentions), dim=1)
                # print("feat shape: ", feat.shape)
                # 1/0

                # all_feat = torch.cat((feat, bg_feat), dim=1)

                output,_ = classifier(feat.cuda())

                # output,_ = classifier(all_feat.cuda())

                outputs.append(output)



            outputs = sum(outputs)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        _, predict = torch.max(all_outputs, 1)
        
        accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy

def train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path = None):

    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    best_acc = 0.0
    best_model = None

    # upsampler
    upsample_fmap = nn.Upsample(scale_factor=4, mode='nearest')

    patches_masked = 4
    kernel_dim = 17

    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    vits16 = vits16.to(device)

    imagenet_mean=(0.485, 0.456, 0.406)
    imagenet_std=(0.229, 0.224, 0.225)

    normalize = transforms.Compose([transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])

    for param in vits16.parameters():
        param.requires_grad = False

    for iter_num in tqdm.tqdm(range(1, args.max_iter + 1)):
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
    
        data_labeled = iter_labeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        label = data_labeled[1].to(device)

        vit_img_in = data_labeled[0][2].to(device)
        orig_img_in = data_labeled[0][3].to(device)

        # shuffle images
        # patch_size = (56, 56)
        # patch_size = (64, 64)
        # divide the batch of images into non-overlapping patches
        # u = nnf.unfold(img_labeled_q, kernel_size=patch_size, stride=patch_size, padding=0)

        # print("featmap cam: ", featmapcam.shape)
        # print("u.shpae: ", u.shape)
        # print("u[0] shape: ", u[0].shape)
        
        # permuted_order = torch.randperm(u[0].shape[-1])

        # mask_patch = torch.randint(u[0].shape[-1], (u[0].shape[0], patches_masked))
        # print("mask patch: ", mask_patch)


        # for i, b_ in enumerate(u):
            # print("mask patch i : ", mask_patch[i].item())
            # print("b_: ", b_)
            # for j in range(patches_masked):
            #     u[i, :, mask_patch[i][j].item()] = torch.zeros(u.shape[1])
                # u[i, :, mask_patch[i][1].item()] = torch.zeros(u.shape[1])
                # u[i, :, mask_patch[i][2].item()] = torch.zeros(u.shape[1])
                # u[i, :, mask_patch[i][3].item()] = torch.zeros(u.shape[1])

        # u[:, :, mask_patch] = torch.zeros((u.shape[0], u.shape[1]))
        # print("permuted_order: ", permuted_order)

        # permute the patches of each image in the batch
        # pu = torch.cat([b_[:, permuted_order][None,...] for b_ in u], dim=0)
        # fold the permuted patches back together
        # f = nnf.fold(pu, img_labeled_q.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)

        # masked_img_labeled_q = nnf.fold(u, img_labeled_q.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)

        # print("masked_img_labeleq_q: ", masked_img_labeled_q.shape)
        
        
        # print("f shape; ", f.shape)

        # 1/0   

        # f = nnf.interpolate(img_labeled_q,scale_factor=0.3,mode='bilinear',align_corners=True) 
    

        # print("img_labeled_q: ", img_labeled_q.shape)
        # print("f shape: ", f.shape)

        # blur = GaussianBlur(kernel_dim)
        # f = blur(img_labeled_q)
        # f = gaussian_blur(img_labeled_q, kernel_dim)

        # print("f shape: ", f.shape)

        # save_image(img_labeled_q[0], 'example_img0.png')
        # save_image(img_labeled_q[1], 'example_img1.png')
        # save_image(img_labeled_q[2], 'example_img2.png')
        # save_image(img_labeled_q[3], 'example_img3.png')
        # save_image(img_labeled_q[4], 'example_img4.png')
        # 1/0

        # save_image(f[0], 'example_img0.png')
        # save_image(f[1], 'example_img1.png')
        # save_image(f[2], 'example_img2.png')
        # save_image(f[3], 'example_img3.png')
        # save_image(f[4], 'example_img4.png')
        # 1/0

        # print("vit_img in: ", vit_img_in.shape)
        
        attentions, foreground, background = get_attentions(vits16, vit_img_in, img_labeled_q, orig_img_in)

        foreground = normalize(foreground)
        # background = normalize(background)


        # 1/0

        # attentions = vits16.get_last_selfattention(vit_img_in)

        # # print("attentions shape: ", attentions.shape)

        # bs = attentions.shape[0]
        # nh = attentions.shape[1] # number of head
        # # we keep only the output patch attention
        # attentions = attentions[:, :, 0, 1:].reshape(bs, nh, -1)

        # w_featmap, h_featmap = 14, 14
        # attentions = attentions.reshape(bs, nh, w_featmap, h_featmap)

        # # print("attentions shape: ", attentions.shape)

        # # attentions = nn.functional.interpolate(attentions, scale_factor=16, mode="nearest").detach().cpu()
        # # attentions = nn.functional.interpolate(attentions, scale_factor=0.5, mode="nearest").detach()
        # m = nn.AvgPool2d(2, stride=2)
        # attentions = m(attentions)


        # print("attnetions shape 2: ", attentions.shape)

        # save_image(vit_img_in[0], 'vit_img_in.png')

        # for j in range(nh):
        #     save_image(attentions[0][j], f"attention_head{j}.png")

        # 1/0



        # featcov16 is the predicted attention map
        # feat_labeled is the pooled output of last conv layer
        # featmap_q is the output of last conv layer

        # feat_labeled, featmap_q, featcov16, bp_out_feat, network = model(img_labeled_q)


        img_labeled_q = foreground.to(device)
        # bg_img_labeled_q = background.to(device)

        feat_labeled, featmap_q, featcov16, bp_out_feat, network = model(img_labeled_q)

        # bg_feat_labeled, bg_featmap_q, bg_featcov16, bg_bp_out_feat, bg_network = model(bg_img_labeled_q)

        # print("feat_labeled shape : ", feat_labeled.shape)
        # print("featmap q shape :", featmap_q.shape)
        # print("attentions shape: ", attentions.shape)

        # attentions = torch.mean(attentions, dim=1)[:, None, :]
        # print("attention maps size: ", attentions.shape)


        # attentions = torch.tile(attentions, (1, 2048, 1, 1))

        # assert torch.equal(attentions[4, 0, :, :], attentions[4, 1, :, :])
        # assert torch.equal(attentions[4, 0, :, :], attentions[4, 2034, :, :])
        # print("attention maps size: ", attentions.shape)

        # hard_attention_maps = torch.mul(attentions, featmap_q)

        # print("hard attention maps: ", hard_attention_maps.shape)

        # avgpool = nn.AvgPool2d(7, stride=1)

        # pooled_features = avgpool(hard_attention_maps)
        # pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # print("pooled features shape: ", pooled_features.shape)
        
        # 1/0

        # print("feat_labeled features: ", feat_labeled.shape)
        

        # print("attentions shape: ", attentions.shape)


        # concat_feats = torch.cat((feat_labeled, attentions), dim=1)
        # print("concat features: ", concat_feats.shape)
        # 1/0



        # print("featmap_q: ", featmap_q.shape)
        # print("feat_labeled shape: ", feat_labeled.shape)
        # print("bg_feat_labeled shape: ", bg_feat_labeled.shape)
        # 1/0
        
        # attended_feats = torch.mul(featmap_q, attentions)

        # shuffled patch output
        # _, _, shuffled_featcov16, _, _ = model(shuffled_img_labeled_q)


        # fg_bg_feats = torch.cat((feat_labeled, bg_feat_labeled), dim=1)
        # print("feat_labeled: ", feat_labeled.shape)
        # print("bg_feat_labeled: ", bg_feat_labeled.shape)
        # print("fg bg feats: ", fg_bg_feats.shape)
        # 1/0

        out, cam_weight = classifier(feat_labeled.cuda()) #feat_labeled/bp_out_feat
        # out, cam_weight = classifier(concat_feats)
        # out, cam_weight = classifier(fg_bg_feats.cuda())

        # out is the classification logits

        #CAM

        weight = cam_weight[label,:]
        weight = weight.to(device)
        weight = weight[:, :, None, None]
        weight_cam = weight.repeat(1, 1, 7, 7)  


        # print("cam_weight: ", cam_weight.shape)
        # print("weight_shape: ", weight.shape)
        # print("weight_cam shape: ", weight_cam.shape)

        #----------------------------------------
        
        classifier_loss = criterions['CrossEntropy'](out, label)

        
        methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}
	
        modelcam = resnet50(projector_dim=args.projector_dim)
        for paramback, paramcam in zip(network.parameters(), modelcam.parameters()):
            paramcam.data.copy_(paramback.data)
            
        target_layers = [modelcam.layer4[-1]]
        input_tensor_labeled_q = img_labeled_q

        target_category = label

        #GradCAM
        cam_algorithm = methods[args.method] 

        with cam_algorithm(model=modelcam,
                        target_layers=target_layers,
                        use_cuda=args.use_cuda) as cam:
        
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 24
        
            # grayscale_cam, weights_gradcam = cam(input_tensor=input_tensor_labeled_q,
            #                     target_category=target_category,
            #                     aug_smooth=args.aug_smooth,
            #                     eigen_smooth=args.eigen_smooth)
            

            # weight_gradcam/weight_cam

            # featmapcam = featmap_q*weight_cam
            featmapcam = torch.rand(weight_cam.shape).cuda()

            # print("featmap camp shape: ", featmapcam.shape)

            featmapcam = torch.sum(featmapcam, dim=1)

            # print("featmap camp shape 2: ", featmapcam[:, None, :, :].shape)

            upsampled_featmapcam = upsample_fmap(featmapcam[:, None, :, :])

            relu = nn.ReLU(inplace=True)
            featmapcam = relu(featmapcam)

            # print("featcov16 shape: ", featcov16.shape)
            

            predictcam, _ = torch.max(featcov16, dim=1)
            predict_cam = predictcam.to(device)

            # print("predict_cam shape: ", predict_cam.shape)
            

            #temperature parameter in softmax
            t = 0.4
            featmapcam = featmapcam.view(featmapcam.size(0), -1)
            featmapcam = (featmapcam/t).float()
            featmapcam = featmapcam.detach()
            predict_cam = predict_cam.view(predict_cam.size(0),-1)
            predict_cam = (predict_cam/t).float()


            loss_cam_labeled_q = F.kl_div(predict_cam.softmax(dim=-1).log(), featmapcam.softmax(dim=-1), reduction='sum')


            # Equivariant regularization loss 
            # try augmentations 

            # PATCH SHUFFLING
            
            # shuffle images
            # patch_size = (7, 7)
            # # divide the batch of images into non-overlapping patches
            # u = nnf.unfold(upsampled_featmapcam, kernel_size=patch_size, stride=patch_size, padding=0)

            # # print("featmap cam: ", featmapcam.shape)
            # # print("u.shpae: ", u.shape)
            # # print("u[0] shape: ", u[0].shape)

            # # print("permuted_order 2: ", permuted_order)

            # # permute the patches of each image in the batch
            # pu = torch.cat([b_[:, permuted_order][None,...] for b_ in u], dim=0)
            # # fold the permuted patches back together
            # shuffled_featmapcam = nnf.fold(pu, upsampled_featmapcam.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
                

            # # print("shuffled featmapcam: ", shuffled_featmapcam.shape)
            # # 1/0

            # shuffled_featmapcam = relu(shuffled_featmapcam[:, 0, :, :])
            # shuffled_featmapcam = shuffled_featmapcam.view(shuffled_featmapcam.size(0), -1)
            # shuffled_featmapcam = (shuffled_featmapcam/t).float()
            # shuffled_featmapcam = shuffled_featmapcam.detach()

            # shuffled_predictcam, _ = torch.max(shuffled_featcov16, dim=1)
            # shuffled_predictcam = upsample_fmap(shuffled_predictcam[:, None, :, :])
            # shuffled_predictcam = shuffled_predictcam[:, 0, :, :]
            # shuffled_predict_cam = shuffled_predictcam.to(device)

            # # print("predict_cam shape: ", predict_cam.shape)
            

            # #temperature parameter in softmax
            # shuffled_predict_cam = shuffled_predict_cam.view(shuffled_predict_cam.size(0),-1)
            # shuffled_predict_cam = (shuffled_predict_cam/t).float()

            # shuffled_loss_cam_labeled_q = F.kl_div(shuffled_predict_cam.softmax(dim=-1).log(), shuffled_featmapcam.softmax(dim=-1), reduction='sum')

        
        # print("hit")
        # total_loss = classifier_loss + 0.01*loss_cam_labeled_q + 0.01*shuffled_loss_cam_labeled_q
        total_loss = classifier_loss + 0.01*loss_cam_labeled_q 
        # total_loss = classifier_loss

        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        # if iter_num % 500 == 0:
        #     patches_masked -= 1
        #     print("patches masked: ", patches_masked)


        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label).sum().item()
            sample_num = predict.size(0)
            print("iter_num: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/cam_loss', loss_cam_labeled_q, iter_num)
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)
        
        # uncomment for testing
        if iter_num % args.test_interval == 1 or iter_num == 500 or iter_num % 100 == 0:
            model.eval()
            classifier.eval()
            test_acc = test(dataset_loaders, model, classifier, device=device)
            print("iter_num: {}; test_acc: {}".format(iter_num, test_acc))
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = {'model': model.state_dict(),
                            'classifier': classifier.state_dict(),
                            'step': iter_num,
                            'best_acc': best_acc
                            }
                
                torch.save(best_model, model_path)
                print("Model saved!")

    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--seed', type=int, default='666666')
    parser.add_argument('--workers', type=int, default='4')
    parser.add_argument('--lr_ratio', type=float, default='10')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum hyperparameter')
    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--max_iter', type=float, default=30000)
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--pretrained_path", type=str, default='~/.torch/models/moco_v2_800ep_pretrain.pth.tar')

    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/test.jpg',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')
    

    configs = parser.parse_args()
    configs.use_cuda = configs.use_cuda and torch.cuda.is_available()
    if configs.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = read_config()
    set_seed(args.seed)

    # Prepare data
    if 'CUB200' in args.root:
        args.class_num = 200
    elif 'StanfordCars' in args.root:
        args.class_num = 196
    elif 'Aircraft' in args.root:
        args.class_num = 100

    dataset_loaders = load_data(args)
    print("class_num: ", args.class_num)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')


    model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.label_ratio))

    logdir = os.path.join(args.logdir, model_name)
    method_name = 'SAM_logConfid'
    method_name += '_qdim' + str(args.projector_dim)

    method_name += '_seed' + str(args.seed)

    logdir = os.path.join(args.logdir, method_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # assert args.pretrained == False

    # Initialize model
    network, feature_dim = load_network(args.backbone)
    model = SAM(network=network, backbone=args.backbone, projector_dim=args.projector_dim,
                       class_num=args.class_num, pretrained=args.pretrained, pretrained_path=args.pretrained_path).to(device)
    classifier = Classifier(2048, args.class_num).to(device)   #2048/num of bilinear 2048*16
    # classifier = Classifier(2342, args.class_num).to(device)
    # classifier = Classifier(4096, args.class_num).to(device)
    
    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))

    ## Define Optimizer
    optimizer = optim.SGD([
        {'params': model.parameters()},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},

    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000, 30000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    train(args, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer, model_path=model_path)

   
if __name__ == '__main__':
    main()
