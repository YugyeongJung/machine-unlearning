# %pip install pillow
# %pip install matplotlib
# %pip install numpy
# %pip install pandas
# %pip install torch
# %pip install torchvision
# %pip install scikit-learn
from flask import Flask # Flask
from flask_cors import CORS
from flask import request
import time
import os
import time
import random
import glob
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import linear_model, model_selection
from PIL import Image
from datetime import datetime

from miface import MI_FACE
from torchvision import models
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

#functions
def parsing(meta_data):
    image_age_list = []
    # iterate all rows in the metadata file
    for idx, row in meta_data.iterrows():
        image_path = row["image_path"]
        age_class = row["age_class"]
        image_age_list.append([image_path, age_class])
    return image_age_list

def train():
    start_time = time.time()
    print(f"[Epoch: {epoch + 1} - Training]")
    model.train()
    total = 0
    running_loss = 0.0
    running_corrects = 0

    for i, batch in enumerate(train_dataloader):
        imgs, labels = batch
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            imgs, labels = imgs.cuda(), labels.cuda()

        outputs = model(imgs)
        optimizer.zero_grad()
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total += labels.shape[0]
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        if i % log_step == log_step - 1:
            print(
                f"[Batch: {i + 1}] running train loss: {running_loss / total}, running train accuracy: {running_corrects / total}"
            )

    print(f"train loss: {running_loss / total}, accuracy: {running_corrects / total}")
    print("elapsed time:", time.time() - start_time)
    return running_loss / total, (running_corrects / total).item()

def test():
    start_time = time.time()
    print(f"[Test]")
    model.eval()
    total = 0
    running_loss = 0.0
    running_corrects = 0

    for i, batch in enumerate(test_dataloader):
        imgs, labels = batch
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            imgs, labels = imgs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        total += labels.shape[0]
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data)

        if (i == 0) or (i % log_step == log_step - 1):
            print(
                f"[Batch: {i + 1}] running test loss: {running_loss / total}, running test accuracy: {running_corrects / total}"
            )

    print(f"test loss: {running_loss / total}, accuracy: {running_corrects / total}")
    print("elapsed time:", time.time() - start_time)
    return running_loss / total, (running_corrects / total).item()

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate
    if epoch >= 10:
        lr /= 10
    if epoch >= 20:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

@torch.no_grad()
def evaluation(model, data_loader):
    start_time = time.time()
    print(f"[Test]")
    model.eval()
    total = 0
    running_loss = 0.0
    running_corrects = 0
    running_top2_corrects = 0
    log_step = 20

    for i, batch in enumerate(data_loader):
        imgs, labels = batch
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            imgs, labels = imgs.cuda(), labels.cuda()
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            # Top-2 accuracy.
            _, top2_preds = outputs.topk(2, dim=1)  # Get the top 2 class indices.
            top2_correct = top2_preds.eq(labels.view(-1, 1).expand_as(top2_preds))
            running_top2_corrects += top2_correct.any(dim=1).sum().item()

        total += labels.shape[0]
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).item()

        if (i == 0) or (i % log_step == log_step - 1):
            """print(f'[Batch: {i + 1}] running test loss: {running_loss / total}, running test accuracy: {running_corrects / total}, running top-2 accuracy: {running_top2_corrects / total}')

            print(f'test loss: {running_loss / total}, accuracy: {running_corrects / total}, top-2 accuracy: {running_top2_corrects / total}')
            print("elapsed time:", time.time() - start_time)"""
    return {
        "Loss": running_loss / total,
        "Acc": running_corrects / total,
        "Top-2 Acc": running_top2_corrects / total,
    }

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_losses(net, loader):
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    use_cuda = torch.cuda.is_available()

    for inputs, y in loader:
        targets = y
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        logits = net(inputs)

        losses = criterion(logits, targets).cpu().detach().numpy()
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def cal_mia(model, forget_dataloader_test, unseen_dataloader):
    set_seed(42)

    forget_losses = compute_losses(model, forget_dataloader_test)
    unseen_losses = compute_losses(model, unseen_dataloader)

    np.random.shuffle(forget_losses)
    forget_losses = forget_losses[: len(unseen_losses)]

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {"MIA": mia_scores.mean(), "Forgeting Score": forgetting_score}

def miaImage(model, targetLabel):
    input_shape = (1, 3, 128, 128)
    lam = 0.03
    num_itr = 1000

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mi = MI_FACE(
        model,
        input_shape,
        target_label=targetLabel,
        num_itr=num_itr,
        lam=lam,
        log_interval=0,
        device=device,
    )
    x_result_1, log = mi.attack()
    x_result_1 = (x_result_1 - x_result_1.min()) / (x_result_1.max() - x_result_1.min())
    return x_result_1[0].detach().cpu().numpy().transpose(1, 2, 0)

# Modified DATASET class
class Dataset(Dataset):
    # choose how to forget
    
    only_upload_data = False

    # Define the path and label for your new image
    upload_path = "/uploads"
    upload_label = "d"  # replace with the actual label
    upload_file_name = "F0001_AGE_M_45_d1.jpg"  # Replace with your actual file name

    # Create the DataFrame
    upload_dataset = pd.DataFrame({"image_path": [upload_file_name], "age_class": [upload_label]})
    def __init__(
        self,
        meta_data,
        image_directory,
        transform=None,
        forget=False,
        retain=False,
        forget_data=upload_dataset,
        only_upload_forget=only_upload_data,

    ):
        self.meta_data = meta_data
        self.image_directory = image_directory
        self.transform = transform

        # Process the metadata.
        image_age_list = parsing(meta_data)
        forget_list = parsing(forget_data)

        self.forget_list = forget_list
        self.image_age_list = image_age_list
        self.age_class_to_label = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
        }

        # After training the original model, we will do "machine unlearning".
        # The machine unlearning requires two datasets, ① forget dataset and ② retain dataset.
        # In this experiment, we set the first 1,500 images to be forgotten and the rest images to be retained.
        if forget:
            if only_upload_forget:
                self.image_age_list = self.forget_list
            else:
                self.image_age_list = self.image_age_list[:1500] + self.forget_list
        if retain:
            self.image_age_list = self.image_age_list[1500:]

    def __len__(self):
        return len(self.image_age_list)

    def __getitem__(self, idx):
        image_path, age_class = self.image_age_list[idx]
        upload_file_name = "F0001_AGE_M_45_d1.jpg"
        if image_path == upload_file_name:
            # img = Image.open(os.path.join(upload_path, image_path))
            img = Image.open("./src/uploads/image.png")
        else:
            img = Image.open(os.path.join(self.image_directory, image_path))
        label = self.age_class_to_label[age_class]

        if self.transform:
            img = self.transform(img)

        return img, label
    
    
@app.route("/ModelCheck", methods=["GET", "POST"])
def ModelCheck():
    
    fileName = request.json['fileName']

    # result_df = pd.read_csv('./src/result.csv')
    # order = str(len(result_df)-1)
    # if(len(result_df) == 1): order = ""
    # if(len(result_df) > 10): order = 10

    with Image.open("./src/" + "uploads" + "/" + fileName) as im:
        new_image = im.resize((128, 128))
        new_image.save("./src/uploads/image.png")
        # new_image.save("./src/" + "uploads" + order + "/" + "image.png")
        # print("./src/" + "uploads" + order + "/" + "image.png")

    # Define the path and label for your new image
    upload_path = "/uploads"
    upload_label = "d"  # replace with the actual label
    upload_file_name = "F0001_AGE_M_45_d1.jpg"  # Replace with your actual file name

    """Downlaoding datasst need to be done sepereately"""
    """ Downloading trained model also needs to be done seperately"""
    """ Handling uploaded image"""
    train_meta_data_path = (
        "./custom_korean_family_dataset_resolution_128/custom_train_dataset.csv"
    )
    train_meta_data = pd.read_csv(train_meta_data_path)
    train_image_directory = "./custom_korean_family_dataset_resolution_128/train_images"

    test_meta_data_path = (
        "./custom_korean_family_dataset_resolution_128/custom_val_dataset.csv"
    )
    test_meta_data = pd.read_csv(test_meta_data_path)
    test_image_directory = "./custom_korean_family_dataset_resolution_128/val_images"

    unseen_meta_data_path = (
        "./custom_korean_family_dataset_resolution_128/custom_test_dataset.csv"
    )
    unseen_meta_data = pd.read_csv(unseen_meta_data_path)
    unseen_image_directory = "./custom_korean_family_dataset_resolution_128/test_images"
    train_transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    unseen_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    test_dataset = Dataset(test_meta_data, test_image_directory, test_transform, forget_data=pd.DataFrame({"image_path": ["./src/" + "uploads" + "/" + fileName], "age_class": ["d"]}))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    unseen_dataset = Dataset(unseen_meta_data, unseen_image_directory, unseen_transform, forget_data=pd.DataFrame({"image_path": ["./src/" + "uploads" + "/" + fileName], "age_class": ["d"]}))
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=64, shuffle=False)


    learning_rate = 0.01
    log_step = 30

    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    num_original_epochs = 30
    best_test_acc = 0
    best_epoch = 0

    """Modified"""
    # choose how to forget
    only_upload_data = False
    if only_upload_data:
        forget_dataset_train = Dataset(
            train_meta_data, train_image_directory, train_transform, forget=True
        )
        forget_dataloader_train = DataLoader(
            forget_dataset_train, batch_size=1, shuffle=False
        )

        forget_dataset_test = Dataset(
            train_meta_data, train_image_directory, test_transform, forget=True
        )
        forget_dataloader_test = DataLoader(
            forget_dataset_test, batch_size=1, shuffle=False
        )
    else:
        forget_dataset_train = Dataset(
            train_meta_data, train_image_directory, train_transform, forget=True
        )
        forget_dataloader_train = DataLoader(
            forget_dataset_train, batch_size=64, shuffle=True
        )

        forget_dataset_test = Dataset(
            train_meta_data, train_image_directory, test_transform, forget=True
        )
        forget_dataloader_test = DataLoader(
            forget_dataset_test, batch_size=64, shuffle=False
        )

    """<b>Original Model Performance</b>"""
    # print(f'last_checkpoint_epoch_{num_original_epochs}.pth')
    # original_save_path = f'last_checkpoint_epoch_{num_original_epochs}.pth' # If you trian the original model from scratch.
    original_save_path = "./content/pre_trained_last_checkpoint_epoch_30.pth"
    original_model = models.resnet18(pretrained=False)
    num_features = original_model.fc.in_features
    original_model.fc = nn.Linear(num_features, 8)
    original_model.load_state_dict(torch.load(original_save_path, map_location=torch.device('cpu')))
    if use_cuda:
        original_model = original_model.cuda()
    criterion = nn.CrossEntropyLoss()
    test_acc = evaluation(original_model, test_dataloader)
    test_acc
    set_seed(42)

    # Performance
    test_acc = evaluation(original_model, test_dataloader)
    unseen_acc = evaluation(original_model, unseen_dataloader)
    if use_cuda:
        original_mia = cal_mia(original_model.cuda())
    else:
        original_mia = cal_mia(original_model)

    print()
    print(f"Test Acc: {test_acc}")
    print(f"Unseen Acc: {unseen_acc}")
    print(f"MIA: {original_mia}")
    print(f'Final Score: {(test_acc["Acc"] + (1 - abs(original_mia["MIA"] - 0.5) * 2)) / 2}')
    return({"original_mia": original_mia})
    return({"original_mia": 1})


@app.route("/Unlearning", methods=["GET", "POST"])
def Unlearning():

    use_cuda = torch.cuda.is_available()
    """Downlaoding datasst need to be done sepereately"""
    """ Downloading trained model also needs to be done seperately"""
    """ Handling uploaded image"""
    train_meta_data_path = (
        "./custom_korean_family_dataset_resolution_128/custom_train_dataset.csv"
    )
    train_meta_data = pd.read_csv(train_meta_data_path)
    train_image_directory = "./custom_korean_family_dataset_resolution_128/train_images"

    test_meta_data_path = (
        "./custom_korean_family_dataset_resolution_128/custom_val_dataset.csv"
    )
    test_meta_data = pd.read_csv(test_meta_data_path)
    test_image_directory = "./custom_korean_family_dataset_resolution_128/val_images"

    unseen_meta_data_path = (
        "./custom_korean_family_dataset_resolution_128/custom_test_dataset.csv"
    )
    unseen_meta_data = pd.read_csv(unseen_meta_data_path)
    unseen_image_directory = "./custom_korean_family_dataset_resolution_128/test_images"
    train_transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    unseen_transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])

    test_dataset = Dataset(test_meta_data, test_image_directory, test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    unseen_dataset = Dataset(unseen_meta_data, unseen_image_directory, unseen_transform)
    unseen_dataloader = DataLoader(unseen_dataset, batch_size=64, shuffle=False)


    learning_rate = 0.01
    log_step = 30

    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 8)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    num_original_epochs = 30
    best_test_acc = 0
    best_epoch = 0

    """Modified"""
    # choose how to forget
    only_upload_data = False
    if only_upload_data:
        forget_dataset_train = Dataset(
            train_meta_data, train_image_directory, train_transform, forget=True
        )
        forget_dataloader_train = DataLoader(
            forget_dataset_train, batch_size=1, shuffle=False
        )

        forget_dataset_test = Dataset(
            train_meta_data, train_image_directory, test_transform, forget=True
        )
        forget_dataloader_test = DataLoader(
            forget_dataset_test, batch_size=1, shuffle=False
        )
    else:
        forget_dataset_train = Dataset(
            train_meta_data, train_image_directory, train_transform, forget=True
        )
        forget_dataloader_train = DataLoader(
            forget_dataset_train, batch_size=64, shuffle=True
        )

        forget_dataset_test = Dataset(
            train_meta_data, train_image_directory, test_transform, forget=True
        )
        forget_dataloader_test = DataLoader(
            forget_dataset_test, batch_size=64, shuffle=False
        )
    
    retain_dataset_train = Dataset(
        train_meta_data, train_image_directory, train_transform, retain=True
    )
    retain_dataloader_train = DataLoader(retain_dataset_train, batch_size=64, shuffle=True)

    retain_dataset_test = Dataset(
        train_meta_data, train_image_directory, test_transform, retain=True
    )
    retain_dataloader_test = DataLoader(retain_dataset_test, batch_size=64, shuffle=False)

    """<b>Original Model Performance</b>"""
    # print(f'last_checkpoint_epoch_{num_original_epochs}.pth')
    # original_save_path = f'last_checkpoint_epoch_{num_original_epochs}.pth' # If you trian the original model from scratch.
    original_save_path = "./content/pre_trained_last_checkpoint_epoch_30.pth"
    original_model = models.resnet18(pretrained=False)
    num_features = original_model.fc.in_features
    original_model.fc = nn.Linear(num_features, 8)
    original_model.load_state_dict(torch.load(original_save_path, map_location=torch.device('cpu')))
    if use_cuda:
        original_model = original_model.cuda()
    criterion = nn.CrossEntropyLoss()
    test_acc = evaluation(original_model, test_dataloader)
    test_acc
    set_seed(42)

    # Performance
    test_acc = evaluation(original_model, test_dataloader)
    unseen_acc = evaluation(original_model, unseen_dataloader)
    if use_cuda:
        original_mia = cal_mia(original_model.cuda())
    else:
        original_mia = cal_mia(original_model)

    print()
    print(f"Test Acc: {test_acc}")
    print(f"Unseen Acc: {unseen_acc}")
    print(f"MIA: {original_mia}")
    print(f'Final Score: {(test_acc["Acc"] + (1 - abs(original_mia["MIA"] - 0.5) * 2)) / 2}')

    """     Negative Gradient Ascent   """
    start = time.time()
    now = datetime.now().strftime('%Y-%m-%d')

    original_save_path = "./content/pre_trained_last_checkpoint_epoch_30.pth"
    unlearned_model = models.resnet18(pretrained=False)
    num_features = unlearned_model.fc.in_features
    unlearned_model.fc = nn.Linear(num_features, 8)
    unlearned_model.load_state_dict(torch.load(original_save_path, map_location=torch.device('cpu')))
    if use_cuda:
        unlearned_model = unlearned_model.cuda()
    criterion = nn.CrossEntropyLoss()

    set_seed(42)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(unlearned_model.parameters(), lr=0.001)

    dataloader_iterator = iter(forget_dataloader_train)

    num_epochs = 2
    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, (x_retain, y_retain) in enumerate(retain_dataloader_train):
            if use_cuda:
                y_retain = y_retain.cuda()

            try:
                (x_forget, y_forget) = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(forget_dataloader_train)
                (x_forget, y_forget) = next(dataloader_iterator)

            if x_forget.size(0) != x_retain.size(0):
                continue

            if use_cuda:
                outputs_forget = unlearned_model(x_forget.cuda())
                loss_ascent_forget = -criterion(outputs_forget, y_forget.cuda())
            else:
                outputs_forget = unlearned_model(x_forget)
                loss_ascent_forget = -criterion(outputs_forget, y_forget)

            optimizer.zero_grad()
            loss_ascent_forget.backward()
            optimizer.step()

            running_loss += loss_ascent_forget.item() * x_retain.size(0)
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(retain_dataloader_train)}] - Batch Loss: {loss_ascent_forget.item():.4f}"
            )

        average_epoch_loss = running_loss / (
            len(retain_dataloader_train) * x_retain.size(0)
        )
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")


    end = time.time()
    total_time = end-start
    # Performance
    test_acc = evaluation(unlearned_model, test_dataloader)
    unseen_acc = evaluation(unlearned_model, unseen_dataloader)
    if use_cuda:
        unlearned_mia = cal_mia(unlearned_model.cuda())
    else:
        unlearned_mia = cal_mia(unlearned_model, forget_dataloader_test, unseen_dataloader)
    print(f"Test Acc: {test_acc}")
    print(f"Unseen Acc: {unseen_acc}")
    print(f"MIA: {unlearned_mia}")
    print(f'Final Score: {(test_acc["Acc"] + (1 - abs(unlearned_mia["MIA"] - 0.5) * 2)) / 2}')


    # result_df = pd.read_csv('./src/result.csv')
    # order = str(len(result_df)-1)
    
    # if(len(result_df) == 1): order = ""
    # if(len(result_df) > 10): order = 10

    original_img = miaImage(original_model, 3)
    original_img_ = Image.fromarray(original_img, "RGB")
    original_img_.save('./src/uploads/original_model.png')
    print('original image saved')

    unlearned_img = miaImage(unlearned_model, 3)
    unlearned_img_ = Image.fromarray(unlearned_img, "RGB")
    unlearned_img_.save('./src/uploads/unlearned_model.png')
    print('unlearned image saved')

    img_difference = np.abs(original_img - unlearned_img)
    img_difference_ = Image.fromarray(img_difference, "RGB")
    img_difference_.save('./src/uploads/difference.png')
    print('difference image saved')

    
    # new_row = {'idx': len(result_df), 'original_mia': original_mia['MIA'], 'unlearned_mia': unlearned_mia['MIA'], 'time': total_time, 'date': now}
    # result_df.loc[len(result_df)] = new_row
    # result_df.to_csv('./src/result.csv')

    """ final return statement original_mia, unlearned_mia"""
    return ({'unlearned_mia': unlearned_mia, 'time': total_time, 'date': now})


if __name__ == "__main__":
    app.run(debug = True)
