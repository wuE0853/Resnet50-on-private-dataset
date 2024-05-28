import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import ResNet50, Bottleneck

from sklearn.metrics import  precision_score, recall_score, f1_score,roc_curve,confusion_matrix

import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "val1": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(p=1),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val2": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val3": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomRotation(90),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val4": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomRotation(90),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val5": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomRotation(180),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val6": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomRotation(180),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val7": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomRotation(270),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val8": transforms.Compose([transforms.Resize((224,224)),
                                     transforms.RandomHorizontalFlip(p=1),
                                     transforms.RandomRotation(270),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
       }
    #log.txt
#     log_path='./log_20240523.txt'
#     logger=open(log_path,'w')
    

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set1", "flower_data1")  # flower data set path
    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    
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


    validate_list=[]
    for t in data_transform:
        validate_list.append(datasets.ImageFolder(root=os.path.join(image_path, "val_888"),
                                            transform=data_transform[t]))
    validate_dataset = torch.utils.data.ConcatDataset(validate_list)
    
    
#     validate_dataset_1 = datasets.ImageFolder(root=os.path.join(image_path, "val"),
#                                             transform=data_transform["val1"])
#     validate_dataset_2 = datasets.ImageFolder(root=os.path.join(image_path, "val"),
#                                             transform=data_transform["val2"])
#     validate_dataset = torch.utils.data.ConcatDataset([validate_dataset_1, validate_dataset_2])
    
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
    print("using {} images for testing.".format(val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    
    # 2. load model
    num_class = 2
    model = ResNet50(Bottleneck,[3,4,6,3], num_class)
    model_weight_path = "./Resnet50_best.pth"
    model.load_state_dict(torch.load(model_weight_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    

    correct=0.0
    total=0.0
    truth=np.array([])
    predict=np.array([])
    score_list=np.array([])
    for batch_idx, (images, labels) in enumerate(validate_loader):
        model.eval()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        sft=nn.Softmax(dim=1)
        p=sft(outputs.data).cpu().numpy()
        score=p[:,1]
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).sum()
        truth=np.hstack((truth,labels.data.cpu().numpy()))
        predict=np.hstack((predict,predicted.data.cpu().numpy()))
        score_list=np.hstack((score_list,score))
        
#     print(truth_list)
#     print(predict_list)
    print('acc is: %.3f%%' % (100 * correct / total))
    print('precision is: %.3f%%' %(100*precision_score(truth,predict,average='binary')))
    print('recall is: %.3f%%' %(100*recall_score(truth,predict,average='binary')))
    print('f1_score is: %.3f%%' %(100*f1_score(truth,predict,average='binary')))
    print('confusion matrix',confusion_matrix(truth,predict))
    
    #roc curve drawing
    fpr, tpr, _ = roc_curve(truth, score_list, pos_label=1)
    plt.plot(fpr, tpr, color='r', label='ROC')
    plt.plot([0, 1], [0, 1], color='b', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('./ROC图.jpg')
            
    
if __name__ == '__main__':
    main()
