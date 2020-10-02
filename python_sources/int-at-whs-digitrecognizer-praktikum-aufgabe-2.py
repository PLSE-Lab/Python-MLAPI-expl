import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

#Download der Trainingsdaten und transformation in gewünschtes Format für die Verarbeitung der Tensoren

train_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, 
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]) ), 
    batch_size=100, shuffle=True)
#Transformation der Trainingsdaten in gewünschtes Format für die Verarbeitung der Tensoren
test_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, download=False, 
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))]) ), 
    batch_size=100, shuffle=True)

#Klasse für Neurolaes Netz
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()               #Konstruktor
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)        #Kompression des Bildes mit Hilfe von Faltung
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)       
        self.conv_dropout = nn.Dropout2d()                  #Dropoutschicht ->"Vergessen" der Daten damit Netzt nicht "auswendig" lernt
        self.fc1 = nn.Linear(320,60)                        #Anzahl der Ein-/Ausgabe der Layer
        self.fc2 = nn.Linear(60, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)                              
        x = F.relu(x)                                       #ReLu-Funktion für die Conv-Schichten
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)                            #Ausgabe mit größten Wert wird auf "1" gesetzt, Rest auf "0"

model = NeuralNetwork()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
def train(epoche):
     model.train()
     for batch_id, (data, target) in enumerate(train_data):
         data = Variable(data)
         target = Variable(target)
         optimizer.zero_grad()
         out = model(data)
         criterion = nn.CrossEntropyLoss()
         loss = criterion(out, target)
         loss.backward()
         optimizer.step()
         print('Epoche: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoche, batch_id*len(data), len(train_data.dataset),
         100.*batch_id/len(train_data), loss.data[0]))



def test():
    model.eval()
    correct = 0
    total = 0
    for data, target in test_data:
        out = model(data)
        _, prediction = torch.max(out.data, 1)
        total += target.size(0)
        correct += (prediction == target).sum().item()
    print('Accuracy: ', 100.*correct/len(test_data.dataset))    



if __name__ == '__main__':
    for epoch in range(1,10):
        train(epoch)
        test()

