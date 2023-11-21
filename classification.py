import torch
from pathlib import Path
import numpy as np
import cv2 as cv
from torchvision import transforms
from tqdm import tqdm
import time
from torchvision.models.resnet import ResNet18_Weights


class Model(torch.nn.Module):
    def __init__(self, num_classes,pretrained=False):
        super().__init__()
        #resnet18 model
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', weights=weights)
        #change the last layer to fit the number of classes
        self.model.fc = torch.nn.Linear(512, num_classes)


    def forward(self, x):
        return self.model(x)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        tqdm.write("Loading data")
        datasetPath = Path("/home/st392/fsl_groups/grp_nlp/compute/RFW/CroppedImages")        
        classes = [d.name for d in datasetPath.iterdir() if d.is_dir()]
        testInclude = np.load(datasetPath.parent/"test.npy", allow_pickle=True).item()
        trainInclude = np.load(datasetPath.parent/"train.npy", allow_pickle=True).item()
        
        #include only the people that are in the train/test split dictionaries
        if self.mode == "train":
            includePeople = trainInclude
        else:
            includePeople = testInclude
        self.transform = transforms.Compose([transforms.ToTensor()])

        for i, c in tqdm(enumerate(classes)):
            includePeopleClass = includePeople[c]
            for image in (datasetPath / c).iterdir():
                user = image.name.split("_")[:-1]
                user = "_".join(user)
                if user not in includePeopleClass:
                    continue
                image = cv.imread(str(image))
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image = self.transform(image)
                self.data.append(image)
                self.labels.append(i)
        tqdm.write("Finished loading data")

    def __getitem__(self, index):
        image_tensor = self.data[index]
        label = self.labels[index]
        return image_tensor, label
    def __len__(self):
        return len(self.data)
def train(pretrained):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = Model(num_classes=4,pretrained=pretrained)
    model.to(device)
    trainDataset = Dataset("train")
    testDataset = Dataset("test")
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=32, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    criterion.to(device)
    savePath = Path("/home/st392/fsl_groups/grp_nlp/compute/RFW/")/"models"/time.strftime("%Y%m%d-%H%M%S_%f")
    try:
        savePath.mkdir(parents=True)
    except:
        time.sleep(1)
        savePath = Path("/home/st392/fsl_groups/grp_nlp/compute/RFW/")/"models"/time.strftime("%Y%m%d-%H%M%S_%f")
        savePath.mkdir(parents=True)

    for epoch in tqdm(range(10)):
        model.train()
        for images, labels in tqdm(trainLoader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        total = 0
        for images, labels in testLoader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        tqdm.write("Accuracy: {}".format(correct/total))
        #save model
        torch.save(model.state_dict(), savePath/f"model_{epoch}_{accuracy}.pt")

if __name__ == "__main__":
    train(pretrained=True)