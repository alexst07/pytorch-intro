
# Preparação dos dados

Cada tipo de camada utilizada pelas redes neurais demandam um tipo diferente de dados. Por exemplo, datasets muito utilizados para redes simples como o Bostom House Prices trabalham com tensores 2D, enquanto imagens coloridas são representadas utilizando tensores 3D.

## Batch

O primeiro eixo(eixo 0) em todos os tensores do PyTorch representa o tensor de amostras, assim imaginando um dataset de imagens, esse eixo representa o número de imagens que serão processadas pelo modelo. Os modelos de deep learning não processam todo o dataset de uma vez, ao invés disso ele quebra esse grande dataset em pequenos conjuntos chamados de batch.
Devido a isso o primeiro eixo é geralmente chamado de *batch axis*.

## Representação de dados no PyTorch


```python
import torch
```

### Tensor 0D ou escalar


```python
x = torch.tensor(55)
print(x)
x.size()
```

    tensor(55)





    torch.Size([])



### Tensor 1D ou vetor

Os tensores de 1 dimensão são semelhante aos vetores ou no caso do python as listas.


```python
x = torch.FloatTensor([4, 6, 8, 10])
print(x)
x.size()
```

    tensor([ 4.,  6.,  8., 10.])





    torch.Size([4])



### Tensor 2D ou matrizes

Dados no formato (samples, features) são representados como matrizes.


```python
x = torch.FloatTensor([[5, 78, 2, 34, 0],
                       [6, 79, 3, 35, 1],
                       [7, 80, 4, 36, 2]])
print(x)
x.size()
```

    tensor([[ 5., 78.,  2., 34.,  0.],
            [ 6., 79.,  3., 35.,  1.],
            [ 7., 80.,  4., 36.,  2.]])





    torch.Size([3, 5])



### Tensor 3D

Os tensores 3D são utilizados para representar dados no formato (samples, timesteps, features). Séries temporais como preço de ações por exemplo são normalmente representada por tensores 3D.


```python
x = torch.FloatTensor([[[5, 78, 2, 34, 0],
                        [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]],
                       [[5, 78, 2, 34, 0],
                        [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]],
                       [[5, 78, 2, 34, 0],
                        [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]]])
print(x)
x.size()
```

    tensor([[[ 5., 78.,  2., 34.,  0.],
             [ 6., 79.,  3., 35.,  1.],
             [ 7., 80.,  4., 36.,  2.]],
    
            [[ 5., 78.,  2., 34.,  0.],
             [ 6., 79.,  3., 35.,  1.],
             [ 7., 80.,  4., 36.,  2.]],
    
            [[ 5., 78.,  2., 34.,  0.],
             [ 6., 79.,  3., 35.,  1.],
             [ 7., 80.,  4., 36.,  2.]]])





    torch.Size([3, 3, 5])



### Tensor 4D

Os tensores 4D são utilizados para representar dados no formato (samples, width, height, channels) ou (samples,
channels, width, height). Esses são os formatos utilizados para armazenar imagens, onde os canais são os 3 canais do RGB.


```python
x = torch.FloatTensor([[[[5, 78, 2, 34, 0],
                        [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]],
                       [[5, 78, 2, 34, 0],
                        [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]],
                       [[5, 78, 2, 34, 0],
                        [6, 79, 3, 35, 1],
                        [7, 80, 4, 36, 2]]]])
print(x)
x.size()
```

    tensor([[[[ 5., 78.,  2., 34.,  0.],
              [ 6., 79.,  3., 35.,  1.],
              [ 7., 80.,  4., 36.,  2.]],
    
             [[ 5., 78.,  2., 34.,  0.],
              [ 6., 79.,  3., 35.,  1.],
              [ 7., 80.,  4., 36.,  2.]],
    
             [[ 5., 78.,  2., 34.,  0.],
              [ 6., 79.,  3., 35.,  1.],
              [ 7., 80.,  4., 36.,  2.]]]])





    torch.Size([1, 3, 3, 5])


