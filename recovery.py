import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from models.cifar100.resnet20 import ClientModel as res20
import tqdm
import inversefed
from statistics import mean 
import os
import torchvision
import datetime
import time
import cv2


DEVICE = 'cuda'

def evaluate(net, dataloader, print_tqdm = True):
      # Define loss function
  criterion = nn.CrossEntropyLoss() # for classification, we use Cross Entropy
  
  with torch.no_grad():
    net = net.to(DEVICE) # this will bring the network to GPU if DEVICE is cuda
    net.train(False) # Set Network to evaluation mode
    running_corrects = 0
    #iterable = tqdm(dataloader) if print_tqdm else dataloader
    iterable = dataloader
    losses = []
    for images, labels in iterable:
      images = images.to(DEVICE, dtype=torch.float)
      labels = labels.to(DEVICE)
      # Forward Pass
      outputs = net(images)
      loss = criterion(outputs, labels)
      losses.append(loss.item())
      # Get predictions
      _, preds = torch.max(outputs.data, 1)
      # Update Corrects
      running_corrects += torch.sum(preds == labels.data).data.item()
    # Calculate Accuracy
    accuracy = running_corrects / float(len(dataloader.dataset))

  return accuracy, mean(losses)



start_time = time.time()
num_images = 1
trained_model = True
target_id = 20
image_path = 'images/'
checkpoint_epochs = 161

setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR100', defs)

model = res20(lr=0.1, num_classes=100, device='cuda')
model.to(**setup)

if trained_model:
    checkpoint = torch.load(f'./checkpoint/resnet20_{checkpoint_epochs}')
    model.load_state_dict(checkpoint['state_dict'])

model.eval();

accuracy = evaluate(model, validloader)[0]
print('\nTest Accuracy: {}'.format(accuracy))


dm = torch.as_tensor(inversefed.consts.cifar100_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar100_std, **setup)[:, None, None]



if num_images == 1:
    if target_id == -1:  # demo image

        ground_truth = torch.as_tensor(
            np.array(Image.open("auto.jpg").resize((32, 32), Image.BICUBIC)) / 255, **setup
        )
        ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()
        
        labels = torch.as_tensor((1,), device=setup["device"])
        target_id = -1
    else:
        #If the target is None take a random image else take id img

        if target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = target_id
        ground_truth, labels = validloader.dataset[target_id]

        ground_truth, labels = (
            ground_truth.unsqueeze(0).to(**setup),
            torch.as_tensor((labels,), device=setup["device"]),
        )
else:
    ground_truth, labels = [], []
    idx = 25 # choosen randomly ... just whatever you want
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))
    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)



model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]

config = dict(signed=False,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.13,
              optim='adam',
              restarts=2,
              max_iterations=8000,
              total_variation=1e-2,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)

#wandb.init(entity = "aml-2022", project="imageReconstruction")
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 32, 32))

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()  
test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1/ds)


os.makedirs(image_path, exist_ok=True)

output_denormalized = torch.clamp(output * ds + dm, 0, 1)
gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)





if num_images == 1:
    rec_filename = f"res20_{target_id}.png"
    torchvision.utils.save_image(output_denormalized, os.path.join(image_path, rec_filename))

    gt_filename = f"groundTruth-{target_id}.png"
    torchvision.utils.save_image(gt_denormalized, os.path.join(image_path, gt_filename))

    fig, ax = plt.subplots(1, 2)
    ax = ax.ravel()
    ax[0].imshow(gt_denormalized[0].cpu().permute(1, 2, 0))
    ax[1].imshow(output_denormalized[0].cpu().permute(1, 2, 0))
    ax[1].set_title(f"loss: {round(stats['opt'],2)} | PSNR: {round(test_psnr, 2)} \n  MSE: {test_mse:2.2f} | FMSE: {feat_mse:2.2e} ", fontsize = 10)

    fig.suptitle(f'resnet20 with {checkpoint_epochs} epochs, img {target_id}', fontsize=13)
    fig.savefig('single_comparison.png')


else:
    fig_gt, ax_gt = plt.subplots(1, num_images, figsize=(7, 7), sharey = True)
    ax_gt = ax_gt.ravel()

    fig_rec, ax_rec = plt.subplots(1, num_images, figsize=(7, 7), sharey = True)
    ax_rec = ax_rec.ravel()

    for idx, img in enumerate(output_denormalized):
        rec_filename = f"res20_{labels[idx]}.png"
        torchvision.utils.save_image(img, os.path.join(image_path, rec_filename))

        gt_filename = f"groundTruth-{labels[idx]}.png"
        torchvision.utils.save_image(gt_denormalized[idx], os.path.join(image_path, gt_filename))

        ax_rec[idx].imshow(img.cpu().permute(1, 2, 0))
        ax_rec[idx].axis("off")

        ax_gt[idx].imshow(gt_denormalized[idx].cpu().permute(1, 2, 0))
        ax_gt[idx].axis("off")
    
    fig_gt.savefig('gt_mlt_comparison.png')
    fig_rec.savefig('rec_mlt_comparison.png')




print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")

 # Print final timestamp
print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
print("---------------------------------------------------")
print(f"Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}")
print("-------------Job finished.-------------------------")