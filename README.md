Беременность - критический период в жизни женщины, и мониторинг здоровья матери является важным для обеспечения благополучия как матери, так и нерожденного ребенка. Этот проект сосредоточен на использовании методов машинного обучения для прогнозирования уровня риска для беременных женщин на основе ключевых атрибутов здоровья. <br>
# Описание данных: <br>
# Age: 
Возраст в годах, когда женщина беременна <br>
# BodyTemp: 
Температура тела, измеряемая в фаренгейтах <br>
# SystolicBP:
Верхнее значение артериального давления в мм рт. ст., другой значимый атрибут во время беременности <br>
# DiastolicBP:
Нижнее значение артериального давления в мм рт. ст., другой значимый атрибут во время беременности <br>
# BS:
Уровень глюкозы в крови, измеряемый молярной концентрацией, ммоль/л <br>
# HeartRate:
Нормальная частота сердечных сокращений в минуту <br>
# Risk Level:
Предсказанный уровень риска во время беременности, учитывая предыдущие атрибуты <br>

# Клонировать репозиторий
git clone https://github.com/Joktun/Pregnancy-Risk-Levels

# Команды для установки библиотек
pip install shap
pip install catboost

# Список необходимых библиотек и их импорт
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score 
import shap

