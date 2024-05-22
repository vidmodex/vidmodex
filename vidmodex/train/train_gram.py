import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision
import os
from vidmodex.tests import myprint
from vidmodex.loss import student_loss
from vidmodex.optim.approximate_gradients import estimate_gradient_objective, compute_grad_norms, measure_true_grad_norm


def train_datafreeGAN(config_args, teacher, teacher_transform, student, student_transform, generator, batch, ref_generator, aux_loss, device, dtype, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""

    teacher.eval()
    student.train()


    optimizer_S, optimizer_G = optimizer

    gradients = []
    cls_name, cls_ind, cls_id = batch
    for i in range(config_args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        # for j, (cls_name, cls_ind, cls_id) in enumerate(dataloader):
        for _ in range(config_args.g_iter):
            # ToDo: Iterate over hyper classes for ref_generator
                        
            z = torch.randn((config_args.batch_size_train, config_args.nz-config_args.n_hyper_class), device=device)
            optimizer_G.zero_grad()
            generator.train()
            # print("start:",cls_name,"end")
            cls_vec = torch.tensor(F.one_hot(cls_ind, num_classes=ref_generator.num_classes), dtype=dtype).to(device)
            
            cond_vec = torch.tensor(F.one_hot(cls_id, num_classes=config_args.n_hyper_class), dtype=dtype).to(device)
            # ref_imgs = ref_generator.sample(cls_vec)
            fake = generator(torch.concat([cond_vec,z], axis=-1))  

            approx_grad_wrt_x, loss_G = estimate_gradient_objective(config_args, teacher, teacher_transform, student, student_transform, fake,
                                                                    epsilon=config_args.grad_epsilon, m=config_args.grad_m,
                                                                    num_classes=config_args.num_classes,
                                                                    device=device, pre_x=True)
            print("forward done")
            fake.backward(approx_grad_wrt_x)

            optimizer_G.step()
            

            C, H, W = fake.shape[1], fake.shape[3], fake.shape[4]
            ## NCTHW -> -1CHW 
            # fake_frames = fake.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W )
            print("done")
            del fake
            cls_vec = cls_vec.repeat_interleave(config_args.frames_per_video, dim=0)
            inp_vec = torch.concat([cond_vec,z], axis=-1)
            fake_len = inp_vec.shape[0]
            for i in tqdm(range(int(np.ceil(fake_len/config_args.batch_size_gram))), desc="Gen_gram_iter", leave=False):
                optimizer_G.zero_grad()
                fake_frame = generator(inp_vec[config_args.batch_size_gram*i:
                                                    min(fake_len, config_args.batch_size_gram*(i+1)), ...]).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W )

                vec = cls_vec[config_args.frames_per_video * config_args.batch_size_gram * i:
                                                    config_args.frames_per_video * min(fake_len, config_args.batch_size_gram * (i+1)), ... ]
                ref_imgs = ref_generator.sample(vec).detach()
                # print("image gen done")
                loss_G2 = aux_loss(fake_frame, ref_imgs) # ToDo: Fix the loss with ref_generator
                loss_G2.backward()
                # print("gradient compute done")
                optimizer_G.step()
            # print("interleave done")
            # ref_imgs = ref_generator.sample(cls_vec).detach()
            


            if i == 0 and config_args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(config_args, fake)

        for _ in range(config_args.d_iter):
            z = torch.randn((config_args.batch_size_train, config_args.nz-config_args.n_hyper_class), device=device)
            cond_vec = torch.tensor(F.one_hot(cls_id, num_classes=config_args.n_hyper_class), dtype=torch.float32).to(device)
            # ref_imgs = ref_generator.sample(cls_vec)
            fake = generator(torch.concat([cond_vec,z], axis=-1)).detach()
            # fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(fake)

            if config_args.loss == "l1" and config_args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if config_args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif config_args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = student(fake.permute(0,2,1,3,4))

            loss_S = student_loss(config_args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        if i % config_args.log_interval == 0:
            file = open(config_args.log_file, "w")
            myprint('Train Epoch:' + str(epoch) + "[" + str( i) + '/' + str(config_args.epoch_itrs) + "(" + str(100 * float(i) / float(
                config_args.epoch_itrs)) + "%)]\tG_Loss: " + str(loss_G.item()) + "S_loss: " + str(loss_S.item()) , file)
            file.close()
            if i == 0:
                with open(config_args.log_dir + "/loss.csv", "a") as f:
                    f.write("%d,%f,%f\n" % (epoch, loss_G, loss_S))

            if config_args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(
                    generator, student)
                if i == 0:
                    with open(config_args.log_dir + "/norm_grad.csv", "a") as f:
                        f.write("%d,%f,%f,%f\n" %
                                (epoch, G_grad_norm, S_grad_norm, x_true_grad))

        config_args.query_budget -= config_args.cost_per_iteration

        torch.save(generator, "weights/gen_model_%d.pt" % i)
        torch.save(student, "weights/student_model_%d.pt" % i)
        torch.save(teacher,  "weights/teacher_model_%d.pt" % i)

        if config_args.query_budget < config_args.cost_per_iteration:
            return loss_S, loss_G
    return loss_S, loss_G

