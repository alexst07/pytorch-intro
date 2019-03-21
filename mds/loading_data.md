
# Carregando dados

Carregar os dados pode ser um tarefa bastante complexa, muitas vezes os datasets são muito grandes e não cabem na memória de uma só vez, por isso é necessário gerenciar o seu carregamento. Caso seja necessário criar o seu próprio data loader o PyTorch provê uma maneira de fazer tal tarefa.

## Classe Dataset

Todas as classes de dataset customizadas do PyTorch devem seguir o esqueleto abaixo:


```python
from torch.utils.data import Dataset
class FaceLandmarksDataset(Dataset):
    def __init__(self,):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
```

Todas as classes derivadas de Dataset devem implementar pelo menos os métodos `__len__` e `__getitem__`. `__len__` diz a quantidade de dados que existe no dataset e `__getitem__` permite acessar os dados pelo seu índice.

### Construindo nossa classe Dataset

No exemplo abaixo iremos implementar a nossa própria classe de Dataset. Esse é um exemplo muito simples, apenas para servir como demonstração. No exemplo a nossa classe apenas gera uma lista de vetores de 2 elementos aleatórios, esse é o nosso dataset, os métodos `__len__` apenas retorna o tamanho da primeira dimensão. E o método `__getitem__`


```python
própriaimport torch
from torch.utils.data import Dataset
import random

class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        
        # gera 15 vetores de 2 elementos
        self.data = torch.rand(15, 2)
        # gera as 15 elementos contendo classes 0, 1 ou 2
        self.classes = torch.tensor([random.randint(0,2) for i in range(0,150)], dtype=torch.int32)
    
    def __len__(self):
        # retorna o tamanho da primeira dimensão
        return self.data.size(0)
    
    def __getitem__(self, i):
        # retornar o vetor na posição i
        return (self.data[i, :], self.classes[i])
```

Para utilizar a nossa classe MyDataset basta instanciar e chamar os métodos através da sobrecarga de operadores do Python.


```python
dataset = MyDataset()

data, label = dataset[3]

print('data:', data)
print('class:', label)
```

    data: tensor([0.5072, 0.1231])
    class: tensor(1, dtype=torch.int32)


# DataLoaders

O DataLoader permite que os dados possam ser iterados na forma de "mini-batches". Esse tipo de processamento é muito útil para se calcular o gradiente descendente por "mini-batches" de um modelo de redes neurais.
Para criar um DataLoader basta utilizar o objeto criado pela classe MyDataset, e fornecer como parâmetros o tamanho do batch e se os dados devem ser ou não embaralhados.


```python
from torch.utils.data import DataLoader

data_loader = DataLoader(dataset, 
                         batch_size=5,   # tamanho do batch de dados
                         shuffle=False)  # se for True, embaralha os dados no inicio de cada iteração
```

Para iterar sobre o objeto DataLoader é bem simples, pois o PyTorch segue bastante o estilo pythonico.


```python
batch_n = 0
n_samples = 0
for data in data_loader:
    print('batch ', batch_n)
    print('data: ', data[0])
    print('label: ', data[1])
    batch_n += 1
    n_samples += len(data[0])
    
print('Tamanho do DataLoader', len(data_loader))
print('Tamanho do dataset', n_samples)
```

    batch  0
    data:  tensor([[0.2012, 0.1475],
            [0.6322, 0.9168],
            [0.9930, 0.0356],
            [0.5072, 0.1231],
            [0.3262, 0.4918]])
    label:  tensor([1, 0, 1, 1, 0], dtype=torch.int32)
    batch  1
    data:  tensor([[0.7051, 0.0360],
            [0.7875, 0.2994],
            [0.9303, 0.5480],
            [0.9907, 0.2935],
            [0.9449, 0.4181]])
    label:  tensor([0, 1, 2, 2, 2], dtype=torch.int32)
    batch  2
    data:  tensor([[0.8469, 0.6955],
            [0.8023, 0.9058],
            [0.0831, 0.0113],
            [0.8126, 0.1513],
            [0.6415, 0.5586]])
    label:  tensor([1, 2, 1, 2, 0], dtype=torch.int32)
    Tamanho do DataLoader 3
    Tamanho do dataset 15


## Utilizando DataLoaders prontos

Para tarefas comuns como carregar várias imagens e realizar data augmentation o PyTorch possui alguns pacotes que já automatizam essa tarefa.

### MNIST


```python
import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# faz o download do dataset MNIST caso esse ainda não exista
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False)

print ('total trainning batch number: {}'.format(len(train_loader)))
print ('total testing batch number: {}'.format(len(test_loader)))
```

    total trainning batch number: 600
    total testing batch number: 100


Primeiro importamos os pacotes necessários, depois é criado o diretório data caso esse ainda não exista, após isso é criada uma função de transformação para normalizar os dados do mnist, o método `dset.MNIST` é de fato quem irá instanciar o Dataset, nesse caso ele irá armazenar em data, e irá fazer download caso esse ainda não foi feito. Após isso instanciamos o DataLoader de treinamento e de teste, com um batch de tamanho 100.

### Diretório de imagens

Para utilizar um diretório contendo suas próprias imagens é bem simples. Primeiro o diretório deve seguir uma estrutura padrão:
```
data/train/dog/xxx.png
data/train/dog/xxy.png
data/train/dog/xxz.png

data/train/cat/123.png
data/train/cat/nsdf3.png
data/train/cat/asd932_.png
```
Nesse caso dog e cat serão os labels


```python
import torchvision.datasets as dset

data_path = 'data/train/'
train_dataset = dset.ImageFolder(
    root=data_path,
    transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=0,
    shuffle=True
)
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-80-a4a8134a32b6> in <module>()
          4 train_dataset = dset.ImageFolder(
          5     root=data_path,
    ----> 6     transform=transforms.ToTensor()
          7 )
          8 train_loader = torch.utils.data.DataLoader(


    ~/anaconda3/lib/python3.6/site-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader)
        176         super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
        177                                           transform=transform,
    --> 178                                           target_transform=target_transform)
        179         self.imgs = self.samples


    ~/anaconda3/lib/python3.6/site-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform)
         73 
         74     def __init__(self, root, loader, extensions, transform=None, target_transform=None):
    ---> 75         classes, class_to_idx = find_classes(root)
         76         samples = make_dataset(root, class_to_idx, extensions)
         77         if len(samples) == 0:


    ~/anaconda3/lib/python3.6/site-packages/torchvision/datasets/folder.py in find_classes(dir)
         21 
         22 def find_classes(dir):
    ---> 23     classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
         24     classes.sort()
         25     class_to_idx = {classes[i]: i for i in range(len(classes))}


    FileNotFoundError: [Errno 2] No such file or directory: 'data/train/'

