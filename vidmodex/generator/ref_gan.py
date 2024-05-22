# ToDO: Include the pytorch Biggan here
import torch as th
import torchvision.transforms as T
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images, display_in_terminal

class ReferenceGenerator:
    def __init__(self, device, S=64):
        self.model = BigGAN.from_pretrained('biggan-deep-256')
        self.device = device
        self.model.to(device)
        self.truncation = 0.4
        self.num_classes = self.model.config.num_classes
        self.resize = T.Resize((S, S))
    def sample(self, class_vec):
        # print(class_vec.shape)
        n = class_vec.shape[0]
        noise_vector = th.tensor(truncated_noise_sample(truncation=self.truncation, batch_size=n)).to(self.device)
        # print(class_vec, noise_vector)
        with th.no_grad():
            output = self.model(noise_vector, class_vec, self.truncation)
            output = self.resize(output)
        return output

    # def sample(self, class_names):
    #     n = len(class_names)
    #     class_vector = one_hot_from_names(class_names, batch_size=n).to('cuda')
    #     noise_vector = truncated_noise_sample(truncation=self.truncation, batch_size=n).to('cuda')
    #     with th.no_grad():
    #         output = self.model(noise_vector, class_vector, self.truncation)
    #     return output
        # ToDo: Fix  
    def name2vec(self, cls_names):
        # n = len(cls_names)
        return one_hot_from_names(cls_names) #, batch_size=n)