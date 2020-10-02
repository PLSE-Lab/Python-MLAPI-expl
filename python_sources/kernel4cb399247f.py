
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = '/Cat_Dog_data'

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transform = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

device = torch.device('cuda')

model = models.densenet121(pretrained=True)

for param in model.parameters():
  param.require_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device)

epochs = 1
steps = 0
running_loss = 0
print_every = 5

for epoch in range(epochs):
  for inputs, labels in trainloader:
    #increase step for every batch
    steps += 1
    inputs, labels = inputs.to(device), labels.to(device)

    optimizer.zero_grad()

    logits = model(inputs)
    loss = criterion(inputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    #validation of the model every 5 steps
    if steps % print_every == 0:
      #this is to deactivate the dropout layout
      model.eval()
      test_loss = 0
      accuracy = 0

      with torch.no_grad():
        for inputs, labels in testloader:
          inputs, labels = inputs.to(device), labels.to(device)

          logits = model(inputs)
          batch_loss += criterion(logits, labels)

          test_loss += batch_loss

          #calculate the accuracy
          ps = torch.exp(logits)
          _, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

      print(f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {running_loss/print_every:.3f}.. "
            f"Test loss: {test_loss/len(testloader):.3f}.. "
            f"Test accuracy: {accuracy/len(testloader):.3f}")
      running_loss = 0
      model.train()
