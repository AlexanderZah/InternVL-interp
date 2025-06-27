# Реализация метода [vl-interp](https://github.com/nickjiang2378/vl-interp) для интерпретации предсказаний мультимодальных моделей. В качестве базовой модели используется **InternVL 2.5**.

## Setup
### Files
```
git clone https://github.com/AlexanderZah/InternVL-interp.git
cd InternVL-interp
```
### Environment
```
python version >= 3.9
pip install -r requirements_for_kaggle.txt
```
### Demos
```
main.ipynb
```


#### P.S. При локальном запуске потребуется установить дополнительные зависимости, так как код запускался на Kaggle, где некоторые библиотеки уже предустановлены :)
точно нужны:
transformers
torch
torchvision
numpy
matplotlib
pillow