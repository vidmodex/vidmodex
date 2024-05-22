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

def train_datafree(trainer, config_args, teacher, teacher_transform, student, student_transform, generator, device, optimizer, epoch, epoch_iters=None):
    """Main Loop for one epoch of Training Generator and Student"""

    teacher.eval()
    student.train()


    optimizer_S, optimizer_G = optimizer

    if epoch_iters is None:
        epoch_iters = config_args.epoch_itrs

    gradients = []

    for i in range(epoch_iters):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(config_args.g_iter):
            # print("started")
            z = torch.randn((config_args.batch_size_z, config_args.nz), device=device)
            # print("no problem in z")
            optimizer_G.zero_grad()
            generator.train()
            
            fake = generator(z)
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
            fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(fake)

            s_logit = F.softmax(student(student_transform(fake)), dim=1)

            loss_S = student_loss(config_args, s_logit, t_logit)
            trainer.manual_backward(loss_S)
            optimizer_S.step()

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

        config_args.query_budget -= config_args.cost_per_iteration

        torch.save(generator, config_args.log_dir + "/weights/gen_model_%d.pt" % i)
        torch.save(student, config_args.log_dir + "/weights/student_model_%d.pt" % i)
        torch.save(teacher,  config_args.log_dir + "/weights/teacher_model_%d.pt" % i)

        if config_args.query_budget < config_args.cost_per_iteration:
            return loss_S, loss_G
    return loss_S, loss_G

