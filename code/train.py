import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
from dataloader import YelpDataset
# from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.metrics import fbeta_score

def load_transform(csv_file, photo_dir, batch_size):
    # normalize the data and then split into training and testing set;
    train_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train_fulldataset = YelpDataset(csv_file, photo_dir, transform=train_transform)
    dataset_len = len(train_fulldataset)
    train_len = int(0.8*dataset_len)
    val_len = dataset_len - train_len
    train_set, val_set = random_split(train_fulldataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size = batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(val_set, batch_size = batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    return train_loader, test_loader

def train_eval(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25, save_dir = "save/"):
    since = time.time()

    os.makedirs(save_dir, exist_ok=True)

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_ED = 0.0
    best_Fbeta = 0.0
    best_cos_sim = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_ED = 0.
            # running_precision = 0.
            # running_recall = 0.
            # running_f1 = 0.
            running_fbeta = 0.
            running_cos_sim = 0.
            iter_cnt = 0
            total_num = {'train': 0, 'val': 0}

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                total_num[phase] += labels.size()[0]
                iter_cnt += 1

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    preds = torch.sigmoid(outputs).cpu().detach().numpy()
                    loss = criterion(outputs, labels)
                    # print("prediction and target")
                    # print(preds)
                    # print(labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                labels = labels.cpu().detach().numpy()
                # print("labels")
                # print(labels.size)
                # print("preds")
                # print(preds.size)
                running_ED += euclidean_distances(labels, preds)
                running_cos_sim += cosine_similarity(labels, preds)
                preds = (preds > 0.5).astype(int)
                # running_ED += euclidean_distances(labels, preds)
                # running_cos_sim += cosine_similarity(labels, preds)
                # running_precision += precision_score(labels, preds, average='micro')
                # running_recall += recall_score(labels, preds, average='micro')
                # running_f1 += f1_score(labels, preds, average='micro')
                # print(labels.size)
                # print(preds.size)
                # print(labels)
                # print(preds)
                running_fbeta += fbeta_score(labels.flatten(), preds.flatten(), average='macro', beta=0.5)
                # print(type(running_fbeta))
                # running_ED.astype(int)

                # print("F1")
                # print(f1_score(labels, preds, average='micro'))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / total_num[phase]
            # epoch_precision = running_precision / iter_cnt
            # epoch_recall = running_recall / iter_cnt
            # epoch_f1 = running_f1 / iter_cnt
            epoch_ED = running_ED / iter_cnt
            epoch_ED = np.average(epoch_ED)
            epoch_fbeta = running_fbeta / iter_cnt
            epoch_cos_sim = running_cos_sim / iter_cnt
            epoch_cos_sim = np.mean(epoch_cos_sim)
            # print(type(np.average(epoch_ED)))
            # print(epoch_f1)
            # print(type(epoch_f1), type(epoch_ED), type(running_ED), type(running_f1))
            # print(epoch_ED)
            # print('{} F1: {:.4f}'.format(phase, epoch_f1))
            # print('{} ED: {:.4f}'.format(phase,epoch_ED))
            # print(type(epoch_cos_sim), type(epoch_fbeta))
            print('{} Loss: {:.4f}   ED: {:.4f} Fbeta:{:.4f} Cos_sim:{:.4f}'  .format(
                phase, epoch_loss,  epoch_ED, epoch_fbeta, epoch_cos_sim))

            # deep copy the model
            if phase == 'val' and epoch_ED > best_ED:
                best_ED = epoch_ED
                # best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(save_dir,"best_ed_{}_epoch_{}".format(best_ED, epoch))
                torch.save(model.state_dict(), save_path)
            if phase == 'val' and epoch_fbeta > best_Fbeta:
                best_Fbeta = epoch_fbeta
                save_path = os.path.join(save_dir, "best_fbeta_{}_epoch_{}".format(best_Fbeta, epoch))
                torch.save(model.state_dict(), save_path)

            if phase == 'val' and epoch_cos_sim > best_cos_sim:
                best_cos_sim = epoch_cos_sim
                save_path = os.path.join(save_dir, "best_cos_{}_epoch_{}".format(best_cos_sim, epoch))
                torch.save(model.state_dict(), save_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val ED: {:4f}'.format(best_ED))
    print('Best val Fbeta: {:4f}'.format(best_Fbeta))
    print('Best val Cos_sim:{:4f}'.format(best_cos_sim))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    # model = nn.Sequential
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model_ft = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model_ft.parameters(), lr = 0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, test_loader = load_transform("../DATA.csv", "../../training_gallary", 10)
    dataloaders = {'train' : train_loader, 'val' : test_loader}
    train_eval(model, criterion, optimizer, exp_lr_scheduler, dataloaders, device, num_epochs=80)