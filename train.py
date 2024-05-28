import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import ResNet50, Bottleneck


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
       
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    #log.txt
    log_path='./log_resnet50_888_20240526.txt'
    logger=open(log_path,'w')
    

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set1", "flower_data1")  # flower data set path
    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    
#     train_list=[]
#     for t in data_transform:
#         if t!='val':
#             train_list.append(datasets.ImageFolder(root=os.path.join(image_path, "train"),
#                                          transform=data_transform[t]))
#     train_dataset = torch.utils.data.ConcatDataset(train_list)
    
#     train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
#                                           transform=data_transform["train"])
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train_888"),
                                            transform=data_transform["train"])
    train_num = len(train_dataset)

#     # {'daisy':0, 'dandelion':1, 'roses':2}
#     flower_list = train_dataset.class_to_idx
#     cla_dict = dict((val, key) for key, val in flower_list.items())
#     # write dict into json file
#     json_str = json.dumps(cla_dict, indent=2)
#     with open('class_indices.json', 'w') as json_file:
#         json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val_888"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    logger.write("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    
    # 2. load model
    num_class = 2
    model = ResNet50(Bottleneck,[3,4,6,3], num_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #epoch = 30

    model_name = "Resnet50"
    loss_function = nn.CrossEntropyLoss()
    val_acc_list=[]

    epochs = 100
    best_acc = 0.0
    save_path = './{}_latest.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        print("epoch:%d/%d"%(epoch+1,epochs))
        # train
        model.train()
        sum_loss=0.0
        correct=0.0
        total=0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (images,labels) in enumerate(train_bar):
            length=len(train_bar)
            images, labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            #print("loss:"+str(loss))

            # print statistics
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total+=labels.size(0)
            correct += predicted.eq(labels.data).sum()
            print('[epoch:%d, iter:%d] Loss:%.03f Acc_train:%.3f%%' %(epoch+1,(step+1+epoch*length),sum_loss/(step+1),100*correct/total))
            logger.write('[epoch:%d, iter:%d] Loss:%.03f Acc_train:%.3f%% \n' %(epoch+1,(step+1+epoch*length),sum_loss/(step+1),100*correct/total))
            
        print('Waiting Val...')
        correct_v=0.0
        total_v=0.0
        for batch_idx, (images, labels) in enumerate(validate_loader):
            model.eval()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_v += labels.size(0)
            correct_v += predicted.eq(labels.data).sum()
        print('Val\'s ac is: %.3f%%' % (100 * correct_v / total_v))
        logger.write('Val\'s ac is: %.3f%% \n' % (100 * correct_v / total_v))
            
        acc_val = 100 * correct_v / total_v
        val_acc_list.append(acc_val)
 
 
        save_path = './{}_latest.pth'.format(model_name)
        torch.save(model.state_dict(), save_path)
        if acc_val == max(val_acc_list):
            save_path='./{}_best.pth'.format(model_name)
            best_epoch=epoch+1
            torch.save(model.state_dict(), save_path)
            print("save epoch {} model".format(epoch+1))

    print('Finished Training. Best val acc:%.3f%%'%(max(val_acc_list)))
    logger.write('Finished Training. Best val acc:%.3f%%  at %d epoch'%(max(val_acc_list), best_epoch))
    os.close(logger)


if __name__ == '__main__':
    main()

