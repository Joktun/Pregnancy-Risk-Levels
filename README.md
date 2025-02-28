Беременность - критический период в жизни женщины, и мониторинг здоровья матери является важным для обеспечения благополучия как матери, так и нерожденного ребенка. Этот проект сосредоточен на использовании методов машинного обучения для прогнозирования уровня риска для беременных женщин на основе ключевых атрибутов здоровья. <br>

| Параметры | Описание данных |
| --- | --- |
| Age | Возраст в годах, когда женщина беременна |
| BodyTemp | Температура тела, измеряемая в фаренгейтах |
| SystolicBP | Верхнее значение артериального давления в мм рт. ст., другой значимый атрибут во время беременности |
| DiastolicBP | Нижнее значение артериального давления в мм рт. ст., другой значимый атрибут во время беременности |
| BS | Уровень глюкозы в крови, измеряемый молярной концентрацией, ммоль/л |
| HeartRate | Нормальная частота сердечных сокращений в минуту |
| Risk Level | Предсказанный уровень риска во время беременности, учитывая предыдущие атрибуты |


## Ссылка на датасет
https://drive.google.com/file/d/194kkxTvtltiDiqcAHjzIpIXqShymPnHX/view?usp=sharing

## Клонировать репозиторий
git clone https://github.com/Joktun/Pregnancy-Risk-Levels

## Команды для установки библиотек
pip install shap <br>
pip install catboost

## Список необходимых библиотек и их импорт
import pandas as pd <br>
import numpy as np <br>
import matplotlib.pyplot as plt <br>
import seaborn as sns <br>

from catboost import CatBoostClassifier <br>
from sklearn.metrics import confusion_matrix <br>
from sklearn.model_selection import train_test_split <br>
from sklearn.model_selection import GridSearchCV <br>
from sklearn.metrics import accuracy_score, f1_score  <br>
import shap

