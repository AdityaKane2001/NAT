import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from dinats import dinat_s_tiny
from wintome_dinats import wintome_dinat_s_tiny
import numpy as np
from tqdm import tqdm

train_ds = datasets.OxfordIIITPet(
    root="/workspace/datasets/akane/oxford",
    split="trainval",
    download=True,
    transform=transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=1),
        ]
    ),
)

test_ds = datasets.OxfordIIITPet(
    root="/workspace/datasets/akane/oxford",
    split="test",
    download=True,
    transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=0, std=1),
        ]
    ),
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
train_dataloader = DataLoader(train_ds, batch_size=64)
test_dataloader = DataLoader(test_ds, batch_size=64)

model = dinat_s_tiny(pretrained=True)
model.head = nn.Linear(768, 37)

def accuracy(true, pred):
    true = np.array(true)
    pred = np.array(pred)
    acc = np.sum((true == pred).astype(np.float32)) / len(true)
    return acc * 100


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)


def fit(model, train_loader, test_loader=None):
    optim = torch.optim.Adam(params=model.parameters(), lr=5e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()

    best_test_acc = -np.inf

    for epoch in range(10):
        print(f"{epoch}/{9} epochs ")
        train_loss = []
        train_preds = []
        train_labels = []
        for batch in tqdm(train_loader):
            imgs = torch.Tensor(batch[0]).to(device)
            labels = torch.Tensor(batch[1]).to(device)
            scores = model(imgs)

            # loss calculation
            loss = criterion(scores, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_loss.append(loss.detach().cpu().numpy())
            train_labels.append(batch[1])
            train_preds.append(scores.argmax(dim=-1))
        loss = sum(train_loss) / len(train_loss)
        acc = accuracy(
            torch.concat(train_labels, dim=0).cpu(),
            torch.concat(train_preds, dim=0).cpu(),
        )
        print(f"\tTrain\tLoss : {round(loss, 3)}", "\t", f"Accuracy : {round(acc, 3)}")

        if test_loader:
            test_loss, test_acc = test(test_loader)
            if test_acc > best_test_acc:
                patient_epochs = 0
                best_test_acc = test_acc
                print(
                    f"\tCurrent best epoch : {epoch} \t Best test acc. : {round(best_test_acc,3)}"
                )
                # torch.save(self.state_dict(), f"{args.output_dir}/vit_task_{args.tasknum}_best.pt")
            else:
                patient_epochs += 1
        print("--" * 100)


def test(test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        test_loss = []
        test_preds = []
        test_labels = []
        for batch in tqdm(test_loader):
            imgs = torch.Tensor(batch[0]).to(device)
            labels = torch.Tensor(batch[1]).to(device)
            scores = model(imgs)
            loss = criterion(scores, labels)
            test_loss.append(loss.detach().cpu().numpy())
            test_labels.append(batch[1])
            test_preds.append(scores.argmax(dim=-1))
        loss = sum(test_loss) / len(test_loss)
        acc = accuracy(
            torch.concat(test_labels, dim=0).cpu(),
            torch.concat(test_preds, dim=0).cpu(),
        )
        print(f"\tTest:\tLoss : {round(loss, 3)}", "\t", f"Accuracy : {round(acc,3)}")

        return loss, acc


fit(model, train_loader, test_loader)
