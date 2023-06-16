import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim

import os
from PIL import Image
import argparse

import datetime

MY_BATCH_SIZE = 32
MY_MODEL_NAME = "resnet"  # e.g., 'resnet', 'vgg', 'mobilenet', 'custom'
MY_EPOCH = 200
MY_LR = 0.001  # original 0.001
MY_MOMENTUM = 0.9  # original 0.9


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


####################
# If you want to use your own custom model
# Write your code here
####################
class Custom_model(nn.Module):
    def __init__(self):
        super(Custom_model, self).__init__()
        # place your layers
        # CNN, MLP and etc.

    def forward(self, input):
        predicted_label = ""
        # place for your model
        # Input: 3* Width * Height
        # Output: Probability of 50 class label
        return predicted_label


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


####################
# Modify your code here
####################
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 50)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=50, bias=True)
        )
    elif selection == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(
            nn.Linear(in_features=1280, out_features=50, bias=True)
        )
    elif selection == "custom":
        model = Custom_model()
    return model


def train(net1, labeled_loader, optimizer, criterion):
    net1.train()
    # Supervised_training
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        # print("batch_idx:", batch_idx)  # MY
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        ####################
        # Write your Code
        # Model should be optimized based on given "targets"
        ####################
        outputs = model(inputs)  # 모델에 입력 데이터를 전달하여 출력을 얻습니다.
        loss = criterion(outputs, targets)  # 모델의 출력과 실제 타겟 간의 손실을 계산합니다.
        loss.backward()  # 손실을 기반으로 모델의 가중치에 대한 그래디언트를 계산합니다.
        optimizer.step()  # 계산된 그래디언트를 사용하여 옵티마이저를 통해 모델의 가중치를 업데이트합니다.


def test(net, testloader):
    print("Test something...")  # MY
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # print("batch_idx:", batch_idx)  # MY
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total


if __name__ == "__main__":
    print("Main Start...")
    if torch.cuda.is_available():
        print("CUDA is here :) / GPU:" + torch.cuda.get_device_name(0))
    else:
        print("No CUDA :(")
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="False")
    parser.add_argument("--student_abs_path", type=str, default="./")
    args = parser.parse_args()

    if not os.path.exists(
        os.path.join(args.student_abs_path, "logs", "Supervised_Learning")
    ):
        os.makedirs(os.path.join(args.student_abs_path, "logs", "Supervised_Learning"))

    batch_size = MY_BATCH_SIZE
    # Input the number of batch size
    if args.test == "False":
        print("Data transforming...")  # MY
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = CustomDataset(
            root="./data/Supervised_Learning/labeled", transform=train_transform
        )
        labeled_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        dataset = CustomDataset(
            root="./data/Supervised_Learning/val", transform=test_transform
        )
        val_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    else:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    model_name = MY_MODEL_NAME
    # Input model name to use in the model_section class
    # e.g., 'resnet', 'vgg', 'mobilenet', 'custom'

    if torch.cuda.is_available():
        model = model_selection(model_name).cuda()
    else:
        model = model_selection(model_name)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # You may want to write a loader code that loads the model state to continue the learning process
    # Since this learning process may take a while.

    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    epoch = MY_EPOCH
    # Input the number of Epochs
    optimizer = optim.SGD(model.parameters(), lr=MY_LR, momentum=MY_MOMENTUM)
    # Your optimizer here
    # You may want to add a scheduler for your loss

    best_result = 0
    if args.test == "False":
        assert params < 7.0, "Exceed the limit on the number of model parameters"
        print("Number of params:", params)
        print("Training...")  # MY
        print("----------------")  # MY

        with open("output_supervised.txt", "a") as file:
            file.write(str(datetime.datetime.now()) + "\n")
            file.write("Model: " + MY_MODEL_NAME + "\n")
            file.write("Epoch: " + str(MY_EPOCH) + "\n")
            file.write("Batch size: " + str(MY_BATCH_SIZE) + "\n")
            file.write("Learning Rate: " + str(MY_LR) + "\n")
            file.write("Momentum: " + str(MY_MOMENTUM) + "\n")
            file.write("Number of params: " + str(params) + "\n")
            file.write("\n")

        res_for_each_epoch = []  # MY
        for e in range(0, epoch):
            print("epoch:", e)  # MY
            train(model, labeled_loader, optimizer, criterion)
            tmp_res = test(model, val_loader)
            # You can change the saving strategy, but you can't change the file name/path
            # If there's any difference to the file name/path, it will not be evaluated.
            print("{}th performance, res : {}".format(e, tmp_res))
            res_for_each_epoch.append(tmp_res)  # MY
            if best_result < tmp_res:
                best_result = tmp_res
                torch.save(
                    model.state_dict(),
                    os.path.join("./logs", "Supervised_Learning", "best_model.pt"),
                )
        print("Final performance {} - {}".format(e, tmp_res))
        print("Best performance {}".format(best_result))

        with open("output_supervised.txt", "a") as file:
            file.write(str(datetime.datetime.now()) + "\n")
            file.write("Best Res: " + str(best_result) + "\n")
            file.write("Last Res: " + str(tmp_res) + "\n")
            file.write(
                "res_for_each_epoch: ["
                + str(", ".join(list(map(str, res_for_each_epoch))))
                + "]"
                + "\n"
            )
            file.write("------------------------------//" + "\n")

    else:
        # This part is used to evaluate.
        # Do not edit this part!
        dataset = CustomDataset(
            root="/data/23_1_ML_challenge/Supervised_Learning/test",
            transform=test_transform,
        )
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        model.load_state_dict(
            torch.load(
                os.path.join(
                    args.student_abs_path,
                    "logs",
                    "Supervised_Learning",
                    "best_model.pt",
                ),
                map_location=torch.device("cuda"),
            )
        )
        res = test(model, test_loader)
        print(res, " - ", params)
