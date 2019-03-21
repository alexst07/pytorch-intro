
# Tensores

Tensor é a principal estrutura de dados utilizada em um framework de deep learning, simplificadamente podemos dizer que o que o pytorch faz é realizar operações sobre os tensores.
Tensores são um containner para dados, no nosso caso dados numéricos, você pode pensar em tensores como uma generalização de matrizes e vetores, assim, um escalar seria um tensor com 0 eixos, um vetor um tensor de 1 eixo e uma matriz um tensor de 2 eixos.

Os tensores tem 3 atributos principais:
* **Rank**: o número de eixos que o tensor tem, por exemplo, um tensor de 3 dimensões tem rank 3, pois tem três eixos.
* **Shape**: é representado por uma tupla de inteiros que descreve quantas dimensões o tensor possui ao longo de cada eixo, por exemplo, uma matriz de 2 dimensões tem shape (3, 5), isso quer dizer que ela possui 3 linhas e 5 colunas.
* **Type**: é o tipo do tensor, por exemplo, um tensor pode ser composto de inteiro de 32 bits com sinal, sem sinal, pontos flutuante, e assim por diante, para ver a lista dos tipos suportados pelo pytorch acesse o link: https://pytorch.org/docs/stable/tensors.html

### Importando PyTorch


```python
import torch
```

## Criando tensores

O PyTorch utiliza sua propria biblioteca de tensores, porque ele acelera as operações de tensores utilizando GPU, no entanto se você está acostumado com o numpy, converter os tensores do PyTorch para numpy ou vice e versa é muito fácil.

#### A partir de lista do python


```python
torch.tensor([[1., -1.], [1., -1.]])
```




    tensor([[ 1., -1.],
            [ 1., -1.]])




```python
# especifica tipo de dados do tensor
torch.tensor([[1., -1.], [1., -1.]], dtype=torch.int32)
```




    tensor([[ 1, -1],
            [ 1, -1]], dtype=torch.int32)



#### A partir de tensores do numpy


```python
import numpy as np

# tensor do pytorch a partir de um tensor do numpy
torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
```




    tensor([[1, 2, 3],
            [4, 5, 6]])



#### Tensores de números aleatórios


```python
# Tensores com números aleatorios
torch.rand(2,4)
```




    tensor([[0.9491, 0.8010, 0.7085, 0.9585],
            [0.1265, 0.1310, 0.7030, 0.8267]])




```python
# Tensores aleatórios com semente, para reproduzir mesma sequência pseudoaleatória
torch.manual_seed(123456)
torch.rand(2,4)
```




    tensor([[0.5043, 0.8178, 0.4798, 0.9201],
            [0.6819, 0.6900, 0.6925, 0.3804]])



#### Tensores de zeros


```python
torch.zeros([2, 4], dtype=torch.int32)
```




    tensor([[0, 0, 0, 0],
            [0, 0, 0, 0]], dtype=torch.int32)



#### Tensores de uns


```python
torch.ones(2,3)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])



#### Matriz identidade


```python
torch.eye(4)
```




    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])



#### Tensores na gpu


```python
if torch.cuda.is_available():
    cuda0 = torch.device('cuda:0')
    torch.ones([2, 4], dtype=torch.float64, device=cuda0)
```

## Operações com tensores

#### visualizando o shape do tensor


```python
t = torch.ones(2,3)
t.size()
```




    torch.Size([2, 3])



#### Modificando o shape do tensor


```python
t1 = torch.ones(4,4)
print(t1.size())
t2 = t1.view(8,2)
print(t2.size())
t3 = t1.view(1, 4,4)
print(t3.size())
```

    torch.Size([4, 4])
    torch.Size([8, 2])
    torch.Size([1, 4, 4])


Sempre que for modificar o shape de um tensor, observe que a multiplicação do valor da dimensão de cada eixo deve ter sempre o mesmo valor, ou seja
\begin{equation*}4*4 = 8*2 = 1*4*4\end{equation*}

### Conversões entre NumPy e Tensores PyTorch

#### Numpy para PyTorch

<font color='red'>ATENÇÃO</font>: na conversão de tensores do numpy para pytorch existe um detalhe a ser que considerado que são que as funções de rede neurais do PyTorch utilizam o tipo FloatTensor e o numpy utiliza como default o tipo float64, o que faz uma conversão automática para DoubleTensor do PyTorch e consequentemente gerando um erro. Portanto devemos utilizar a função **FloatTensor** para realizar essa conversão.


```python
np_t = np.ones((4,5))
pt_t = torch.FloatTensor(np_t)
pt_t
```




    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])



#### PyTorch para numpy


```python
pt_t = torch.ones(2,3)
np_t = pt_t.numpy()
np_t
```




    array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)



#### Visualização do shape (size)


```python
pt_t = torch.ones(5,3,2)
pt_t.size()
```




    torch.Size([5, 3, 2])



#### Reshape com a função (view)


```python
pt_t = torch.ones(5,3,2)
b = pt_t.view(2,5,3)
b.size()
```




    torch.Size([2, 5, 3])



#### Adição e subtração elemento por elemento


```python
a = torch.arange(0,24).view(4,6)
b = torch.arange(0,24).view(4,6)
c = a + b
c
```




    tensor([[ 0,  2,  4,  6,  8, 10],
            [12, 14, 16, 18, 20, 22],
            [24, 26, 28, 30, 32, 34],
            [36, 38, 40, 42, 44, 46]])




```python
d = a - b
d
```




    tensor([[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])




```python
# forma funcional
e = a.add(b)
e
```




    tensor([[ 0,  2,  4,  6,  8, 10],
            [12, 14, 16, 18, 20, 22],
            [24, 26, 28, 30, 32, 34],
            [36, 38, 40, 42, 44, 46]])




```python
# forma funcional
f = a.sub(b)
f
```




    tensor([[0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]])




```python
# operação inplace
a.add_(b)
a
```




    tensor([[ 0,  2,  4,  6,  8, 10],
            [12, 14, 16, 18, 20, 22],
            [24, 26, 28, 30, 32, 34],
            [36, 38, 40, 42, 44, 46]])




```python
# operação inplace
a.sub_(b)
a
```




    tensor([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]])



#### Muitiplicação elemento por elemento


```python
# usando sobrecarga de operadores
c = a * b
c
```




    tensor([[  0,   1,   4,   9,  16,  25],
            [ 36,  49,  64,  81, 100, 121],
            [144, 169, 196, 225, 256, 289],
            [324, 361, 400, 441, 484, 529]])




```python
# usando chamada de função
d = a.mul(b)
d
```




    tensor([[  0,   1,   4,   9,  16,  25],
            [ 36,  49,  64,  81, 100, 121],
            [144, 169, 196, 225, 256, 289],
            [324, 361, 400, 441, 484, 529]])




```python
# operação inplace
a.mul_(b)
a
```




    tensor([[  0,   1,   4,   9,  16,  25],
            [ 36,  49,  64,  81, 100, 121],
            [144, 169, 196, 225, 256, 289],
            [324, 361, 400, 441, 484, 529]])



#### Divisão por um escalar


```python
c = a/2
c
```




    tensor([[  0,   0,   2,   4,   8,  12],
            [ 18,  24,  32,  40,  50,  60],
            [ 72,  84,  98, 112, 128, 144],
            [162, 180, 200, 220, 242, 264]])



#### Média


```python
# O metodo type é utilizado para converter o tipo do tensor
a = torch.arange(0,20).type(torch.FloatTensor).view(4,5)
a
```




    tensor([[ 0.,  1.,  2.,  3.,  4.],
            [ 5.,  6.,  7.,  8.,  9.],
            [10., 11., 12., 13., 14.],
            [15., 16., 17., 18., 19.]])




```python
# o metodo mean calcula a média do tensor inteiro resultando um escalar
u = a.mean()
u
```




    tensor(9.5000)



#### Média em um eixo


```python
# calcula a média em cada linha, como existem 4 linhas, retorna um vetor de 4 elementos
u_row = a.mean(dim=1)
u_row
```




    tensor([ 2.,  7., 12., 17.])




```python
# calcula a média em cada coluna, como existem 5 colunas, retorna um vetor de 5 elementos
u_col = a.mean(dim=0)
u_col
```




    tensor([ 7.5000,  8.5000,  9.5000, 10.5000, 11.5000])



#### Desvio padrão


```python
std = a.std()
std
```




    tensor(5.9161)




```python
std_row = a.std(dim=1)
std_row
```




    tensor([1.5811, 1.5811, 1.5811, 1.5811])




```python
std_col = a.std(dim=0)
std_col
```




    tensor([6.4550, 6.4550, 6.4550, 6.4550, 6.4550])


