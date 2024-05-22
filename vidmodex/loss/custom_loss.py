import torch
import torch.nn.functional as F

def student_loss(config_args, s_logit, t_logit, return_t_logits=False, reduction=None):
    """Kl/ L1 / CrossEntrophy Loss for student"""
    print_logits = False
    if config_args.loss == "l1":
        loss_fn = F.l1_loss
        if config_args.no_logits:
            t_logit = torch.log(t_logit).detach()
            if config_args.logit_correction == 'min':
                t_logit = t_logit.detach() - t_logit.min(dim=1).values.view(-1, 1).detach()
            elif config_args.logit_correction == 'mean':
                t_logit = t_logit.detach() - t_logit.mean(dim=1).view(-1, 1).detach()
                
        loss = loss_fn(s_logit, t_logit.detach(), reduction=reduction or "mean")
    elif config_args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = torch.log(s_logit)  # log - Probability
        # t_logit = F.softmax(t_logit, dim=1)  # Probality
        loss = loss_fn(s_logit, t_logit.detach(), reduction=reduction or "batchmean")
    elif config_args.loss == "cross_entropy":
        loss_fn = F.nll_loss
        s_logit = torch.log(s_logit)
        loss = loss_fn(s_logit, t_logit.detach().argmax(dim=1), reduction=reduction or "mean") #todo: check
    else:
        raise ValueError(config_args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


def generator_loss(config_args, s_logit, t_logit, z=None, z_logit=None, return_t_logits=False, reduction="mean"):
    """Kl/ L1 / CrossEntrophy Loss for student"""
    if config_args.gen_loss == "l1":
        loss_fn = F.l1_loss
        if config_args.no_logits:
            t_logit = torch.log(t_logit).detach()
            if config_args.logit_correction == 'min':
                t_logit = t_logit.detach() - t_logit.min(dim=1).values.view(-1, 1).detach()
            elif config_args.logit_correction == 'mean':
                t_logit = t_logit.detach() - t_logit.mean(dim=1).view(-1, 1).detach()
                
        loss = loss_fn(s_logit, t_logit.detach(), reduction=reduction)
    elif config_args.gen_loss == "kl":
        loss_fn = F.kl_div
        s_logit = torch.log(s_logit)  # log - Probability
        if config_args.no_prob:
            t_logit = F.softmax(t_logit.detach(), dim=1)  # Probality
        loss = loss_fn(s_logit, t_logit.detach(), reduction=reduction)
    elif config_args.gen_loss == "cross_entropy":
        loss_fn = F.nll_loss
        s_logit = torch.log(s_logit)
        loss = loss_fn(s_logit, t_logit.detach().argmax(dim=1), reduction=reduction) #todo: check
    else:
        raise ValueError(config_args.gen_loss)

    if return_t_logits:
        return - loss, t_logit.detach()
    else:
        return - loss
