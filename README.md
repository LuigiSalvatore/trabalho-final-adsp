# Automatização de Triagem de Infrações de Trânsito: Extração de Velocidade via PDS e OCR

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-PDI-green.svg)
![EasyOCR](https://img.shields.io/badge/EasyOCR-OCR-orange.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

Projeto de Extensão desenvolvido para a disciplina de **Aplicações de Processamento Digital de Sinais (4456S-04)** do Curso de Engenharia de Computação da **Escola Politécnica - PUCRS (2026/1)**, em parceria com o **LABELO PUCRS**.

---

## 📄 Sobre o Projeto

O objetivo deste projeto é automatizar a triagem de fotos de infrações de trânsito emitidas por medidores de velocidade (radares multifabricantes). A aplicação utiliza técnicas de **Processamento Digital de Sinais e Imagens (PDS/PDI)** combinadas com **Reconhecimento Óptico de Caracteres (OCR)** e **Expressões Regulares (Regex)** para detectar e extrair com precisão a velocidade medida registrada nas tarjas de metadados das imagens.

### 🎯 História de Usuário & Critérios de Aceite (LABELO)
* **História de Usuário**: Como usuário de laboratório, quero que o sistema extraia automaticamente a velocidade medida contida nas imagens de infrações, independentemente do fabricante do radar, para automatizar a triagem e reduzir o esforço manual.
* **Compatibilidade Multifabricante**: Suporte a múltiplos layouts de tarjas de metadados (modelos SPL-MFR, SPL-MFS, VSISVCAP01, etc.).
* **Exatidão na Extração**: Isolamento preciso da Velocidade Medida (`Velmed`), diferenciando-a da Velocidade Máxima (`Velmax`) e Considerada (`Velconsid`).
* **Saída Padronizada**: Exportação em arquivo `.csv` estruturado contendo os valores numéricos limpos para posterior cálculo e integração em banco de dados.

---

## 🛠️ Arquitetura e Tecnologia

A solução foi estruturada de forma modular em Python:

```text
Projeto de Extensão/
├── src/
│   ├── __init__.py
│   ├── image_processing.py  # Funções de PDI (Leitura Unicode, ROI, CLAHE, Binarização)
│   ├── speed_extractor.py   # OCR multilíngue e filtragem por Regex multifabricante
│   └── export_results.py    # Amostragem em lote e exportação de relatórios em CSV
├── Fotos_Medidor de velocidade/ # Conjunto de dados de imagens de teste
├── main.py                  # Script CLI principal de execução do pipeline
├── resultados_extracao_velocidade.csv # Tabela de resultados exportada
├── README.md                # Documentação do repositório
└── .gitignore               # Arquivo de exclusões do Git
```

### 🔬 Técnicas de PDS e PDI Aplicadas
1. **Tratamento de Caminhos Unicode**: Carregamento de imagens utilizando `np.frombuffer` e `cv2.imdecode` para garantir compatibilidade com caracteres especiais e acentuação em sistemas Windows.
2. **Recorte de Regiões de Interesse (ROI)**: Isolamento adaptativo das faixas horizontais superiores e inferiores (`crop_metadata_region`) onde situam-se os metadados.
3. **Equalização Adaptativa de Histograma (CLAHE)**: Ajuste local de contraste para realce de caracteres em condições desfavoráveis de iluminação.
4. **Binarização Adaptativa**: Aplicação de thresholding de Otsu/Gaussiano para destacar textos ruidosos sobre fundos escuros ou claros.
5. **Filtragem Numérica por Regex**: Padrões dinâmicos inteligentes capazes de extrair a velocidade medida ignorando variações ruidosas do OCR (ex: `knvh`, `kmlh`, `kmvh`).

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado. Instale as dependências executando:

```bash
pip install opencv-python pillow easyocr pandas numpy
```

### Executando a Extração
Para rodar a extração em lote sobre as amostras e gerar o relatório em CSV, execute no terminal:

```bash
python main.py
```

---

## 📊 Resultados Obtidos

Nos testes automatizados realizados sobre o lote de amostras representativas dos fabricantes A, B e C, o sistema alcançou **100% de precisão na extração** da velocidade medida, salvando os resultados padronizados no arquivo `resultados_extracao_velocidade.csv`.

---

## 👥 Autores e Créditos
* **Disciplina**: Aplicações de Processamento Digital de Sinais (4456S-04)
* **Instituição**: Escola Politécnica - PUCRS
* **Parceiro Institucional**: LABELO PUCRS
