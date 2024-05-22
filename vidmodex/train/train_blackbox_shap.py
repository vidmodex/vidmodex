import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import os
import tqdm
from vidmodex.tests import myprint
from vidmodex.loss import student_loss
from vidmodex.optim.approximate_gradients import estimate_gradient_objective, compute_grad_norms, measure_true_grad_norm

def train_shap_datafree(trainer, config_args, teacher, teacher_transform, student, student_transform, generator, discriminator, shap_loss, cls_id, victim_max_evals, device, optimizer, epoch, epoch_iters=None):
    """Main Loop for one epoch of Training Generator and Student"""
    
    teacher.eval()
    student.train()

    
    optimizer_S, optimizer_G, optimizer_D = optimizer

    if epoch_iters is None:
        epoch_iters = config_args.epoch_itrs

    gradients = []
    cls_ids = torch.repeat_interleave(cls_id, config_args.batch_size_z, dim=0).to(device=device)
    
    shap_loss.set_max_evals(victim_max_evals)
    
    for i in range(epoch_iters):
        """Repeat epoch_itrs times per epoch"""
        cls_idx = cls_ids[i*config_args.batch_size_z:(i+1)*config_args.batch_size_z].reshape(-1)
        for _ in range(config_args.g_iter):
            # print("started")
            z = torch.randn((config_args.batch_size_z, config_args.nz), device=device)
            # print("no problem in z")
            optimizer_G.zero_grad()
            generator.train()
            fake = generator(z, cls_idx)
            # print(fake.shape)  

            approx_grad_wrt_x, loss_G = estimate_gradient_objective(config_args, teacher, teacher_transform, student, student_transform, fake,
                                                                    epsilon=config_args.grad_epsilon, m=config_args.grad_m,
                                                                    num_classes=config_args.num_classes,
                                                                    device=device, pre_x=True)
            trainer.manual_backward(fake, approx_grad_wrt_x)
            optimizer_G.step()

            if i == 0 and config_args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(config_args, teacher, teacher_transform, student, student_transform, device, fake)

        for _ in range(config_args.d_iter):
            z = torch.randn((config_args.batch_size_z, config_args.nz)).to(device)
            fake = generator(z, cls_idx).detach()
            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(fake)

            if config_args.loss == "l1" and config_args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if config_args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif config_args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = student(student_transform(fake))

            loss_S = student_loss(config_args, s_logit, t_logit)
            trainer.manual_backward(loss_S)
            optimizer_S.step()
        
        
        for _ in range(config_args.shap_iter):
            shap_gt_cache=None
            shap_gt_abs_max_val_cache = None
            target_cls_cache = None
            optimizer_G.zero_grad()
            z = torch.randn((config_args.batch_size_z, config_args.nz), device=device)
            fake = generator(z, cls_idx)
            if teacher_transform:
                fake = teacher_transform(fake)
            
            shap_out_mu, shap_out_logvar = discriminator(fake, cls_idx)
            shap_out_sigma = torch.exp(0.5*shap_out_logvar)
            loss_D, (shap_gt, shap_gt_abs_max_val, target_cls) = shap_loss(fake.detach(), shap_out_mu, shap_out_sigma.clip(min=config_args.min_shap_sigma), optimize_prob=False, optimize_energy=True,class_idx=cls_idx, shap_gt=shap_gt_cache)            
            shap_gt_cache = shap_gt if shap_gt is not None else shap_gt_cache
            shap_gt_abs_max_val_cache = shap_gt_abs_max_val if shap_gt_abs_max_val is not None else shap_gt_abs_max_val_cache
            target_cls_cache = target_cls if target_cls is not None else target_cls_cache
            trainer.manual_backward(loss_D)
            optimizer_G.step()
            
            if config_args.optimize_prob and victim_max_evals>0:
                discriminator.train()
                fake = fake.detach()
                for _ in range(config_args.shap_prob_iter):
                    optimizer_D.zero_grad()
                    
                    
                    shap_out_mu, shap_out_logvar = discriminator(fake, cls_idx)
                    shap_out_sigma = torch.exp(0.5*shap_out_logvar)
                    prob_loss, _ = shap_loss(fake, shap_out_mu, shap_out_sigma, optimize_prob=True, optimize_energy=False, class_idx=cls_idx, shap_gt=shap_gt_cache)
                    trainer.manual_backward(prob_loss)
                    optimizer_D.step()
                        
                discriminator.eval()
        
        if i==0 or (i+1) % config_args.log_interval == 0:
            file = open(config_args.log_file,'w')
            file.seek(0,2)
            myprint('Train Epoch:' + str(epoch) + "[" + str(i) + '/'+ str(config_args.epoch_itrs)+ "("+ str(100 * float(i) / float(
                config_args.epoch_itrs) )+ "%)]\tG_Loss: " + str(loss_G.item()) + "S_loss: " + str(loss_S.item()) , file)
            file.close()
            if i == 0:
                with open(config_args.log_dir + "/loss.csv", "a") as f:
                    f.seek(0,2)
                    f.write("%d,%f,%f\n" % (epoch, loss_G, loss_S))

            if config_args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(
                    generator, student)
                if i == 0:
                    with open(config_args.log_dir + "/norm_grad.csv", "a") as f:
                        f.seek(0,2)
                        f.write("%d,%f,%f,%f\n" %
                                (epoch, G_grad_norm, S_grad_norm, x_true_grad))

        config_args.query_budget -= config_args.cost_per_iteration + (config_args.shap_cost_per_epoch_per_eval * victim_max_evals)

        torch.save(generator, config_args.log_dir + "/weights/gen_model_%d.pt" % i)
        torch.save(student, config_args.log_dir + "/weights/student_model_%d.pt" % i)
        torch.save(teacher,  config_args.log_dir + "/weights/teacher_model_%d.pt" % i)

        if config_args.query_budget < config_args.cost_per_iteration:
            return loss_S, loss_G, loss_D, prob_loss
    return loss_S, loss_G, loss_D, prob_loss

