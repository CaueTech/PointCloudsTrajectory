# 3D LiDAR Odometry using Iterative Closest Point (ICP)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositório contém uma implementação modular em Python para estimativa de trajetória e odometria visual 3D utilizando o algoritmo **Iterative Closest Point (ICP)** sobre sequências de nuvens de pontos LiDAR do dataset KITTI.

O objetivo do projeto é alinhar recursivamente scans consecutivos de LiDAR para reconstruir a trajetória percorrida por um veículo e comparar o resultado estimado com a trajetória real de referência (*Ground Truth*), avaliando a precisão via **Root Mean Square Error (RMSE)**.

---

## 📐 Visão Geral da Arquitetura

O código foi refatorado seguindo os princípios de modularização e separação de responsabilidades para facilitar o estudo e manutenção:

```text
.
├── assets/
│   ├── KITTI-Sequence/       # Nuvens de pontos 3D (.obj)
│   └── ground_truth.npy      # Poses de referência relativas e globais
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Carregamento do dataset e pré-processamento de I/O
│   ├── geometry.py           # Operações geométricas (downsampling e transformações)
│   ├── icp.py                # Algoritmo de alinhamento ICP com rejeição de outliers
│   ├── metrics.py            # Cálculo de erros de trajetória (RMSE)
│   └── visualization.py      # Plotagem 3D interativa com Matplotlib
├── main.py                   # Entrypoint e orquestrador do pipeline
├── requirements.txt          # Dependências do ecossistema Python
├── .gitignore
└── README.md
```

---

## 🔬 Como Funciona o Algoritmo (ICP Pipeline)

1. **Voxel Downsampling**: Redução da densidade das nuvens de pontos utilizando uma grade voxel (0.5 m) para otimizar a velocidade de busca.
2. **Busca de Correspondências**: Busca pelos vizinhos mais próximos via **KD-Tree** (`scipy.spatial.KDTree`).
3. **Rejeição de Outliers**: Filtro percentilar que remove os 5% dos pontos mais distantes em cada iteração, mantendo apenas 95% das melhores correspondências.
4. **Estimativa de Rotação e Translação**: Utilização da Decomposição em Valores Singulares (**SVD**) para calcular a matriz de rotação **R** e o vetor de translação **t** ideais.
5. **Acúmulo de Poses**: Integração temporal das transformações relativas para obter a trajetória global estimada.

---

## 🛠️ Pré-requisitos e Instalação

### 1. Requisitos do Sistema Operacional (Linux/Ubuntu)

Para abrir a visualização 3D interativa do Matplotlib, certifique-se de ter a interface gráfica do Tkinter instalada:

```bash
sudo apt update
sudo apt install python3-tk
```

### 2. Configuração do Ambiente Virtual

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalação das Dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Executando o Projeto

```bash
python main.py
```

### Exemplo de Saída

```text
1. Carregando dados do KITTI...
2. Executando alinhamento via ICP...
   Processando scan nº 0...
   Erro ICP para scan 0: 0.1245
   ...
   Erro ICP para scan 28: 0.2104

RMSE (Estimado vs Real): 17.6399 m
3. Exibindo visualização 3D...
```

---

## 📊 Resultados

Ao final da execução, uma janela 3D será aberta mostrando:

- **Linha Azul:** trajetória estimada pelo ICP.
- **Linha Vermelha Tracejada:** trajetória real (*Ground Truth*).

---

## 📜 Licença

Este projeto é disponibilizado para fins acadêmicos e educacionais sob a licença MIT.