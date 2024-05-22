import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def myprint(a, file):
    """Log the print statements"""
    file.write(a); file.write("\n"); file.flush()

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    scores = scores.cpu().numpy()
    labels = labels.cpu().numpy()
    res = {}
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res[f"top_{k}_acc"] = topk_acc_score
    # print(res)
    return res

def epoch_test(args, file, student = None, generator = None, device = "cuda", test_loader = None, epoch=0):
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            # print("data size", data.all()==0.0)
            output = student(data)
            # print("output size", output.shape)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            correct += data.shape[0]*top_k_accuracy(output, target, (5,10,1))[0]
            
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    myprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy) , file)
    with open(args.log_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n"%(epoch, accuracy))
    acc = correct/len(test_loader.dataset)
    return acc, test_loss

def batch_test(args, batch, student = None, generator = None, device = "cuda"):
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = batch
        data, target = data.to(device), target.to(device)

        output = student(data)
        test_loss = F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        accs = top_k_accuracy(output, target, args.val_topk_accuracy)
        correct_dict = {k: data.shape[0]*v for k,v in accs.items()}
            
    accuracy = {k: correct / len(data) for k, correct in correct_dict.items()}
    
    return accuracy, test_loss