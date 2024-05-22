import torch

def shap_prob_test(args, batch, discriminator, shap_loss, device):
    with torch.no_grad():
        data, target = batch
        data, target = data.to(device), target.to(device)
        shap_out_mu, shap_out_logvar = discriminator(data, target)
        shap_out_sigma = torch.exp(0.5*shap_out_logvar)
        prob_loss, _ = shap_loss(data, shap_out_mu, shap_out_sigma, optimize_prob=True, optimize_energy=False, class_idx=target)
    return prob_loss