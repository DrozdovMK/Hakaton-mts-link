# -*- coding: utf-8 -*-
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, QFrame,QFileDialog,QHBoxLayout,
                             QVBoxLayout, QPushButton, QLineEdit, QLabel, QFormLayout, QSizePolicy, QFileDialog)
from scripts.PlotPyQT import BarChartWidget
import numpy as np
from scripts.embedding_model import RuBertEmbedder, universal_sentence_encoder
from scripts.clasterer import Clasterer
import pandas as pd
class UI(QMainWindow):
    def __init__(self):
        super().__init__()
        # self.setGeometry(400, 200, 900, 700)
        self.setWindowTitle('Аналитика ответов')
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        # Создаем два фрейма для каждой части окна, слева и справа
        frame1 = QFrame()
        frame1.setFrameShape(QFrame.StyledPanel)
        frame1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        frame2 = QFrame()
        frame2.setFrameShape(QFrame.StyledPanel)
        frame2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        # Размещаем фреймы в главном окне
        
        frame1_layout = QVBoxLayout() # Вертикальная компоновка 
        frame2_layout = QVBoxLayout() # Вертикальная компоновка 
        frame1.setLayout(frame1_layout)
        frame2.setLayout(frame2_layout)
        main_layout.addWidget(frame1,)
        main_layout.addWidget(frame2,)
        
        frame1_1 = QFrame()
        frame1_1.setFrameShape(QFrame.StyledPanel)
        frame1_2 = QFrame()
        frame1_2.setFrameShape(QFrame.StyledPanel)
        
        frame1_layout.addWidget(frame1_1)
        frame1_layout.addWidget(frame1_2)
        
        frame1_1_layout = QVBoxLayout()
        frame1_2_layout= QVBoxLayout()
        frame1_1.setLayout(frame1_1_layout)
        frame1_2.setLayout(frame1_2_layout)
        
        
        self.app_descr_layout = QVBoxLayout()
        self.app_name = QLabel("Анализ ответов", alignment=QtCore.Qt.AlignCenter)
        
        with open('app_description.txt', 'r', encoding='utf-8') as file:
            # Чтение содержимого файла
            app_description = file.read()
        self.app_description = QLabel(app_description)
        self.app_description.setWordWrap(True)
        self.app_descr_layout.addWidget(self.app_name)
        self.app_descr_layout.addWidget(self.app_description)
        
        self.model_choose_layout = QHBoxLayout() # второй layout
        self.model_1_button = QPushButton("Загрузить RuBERT")
        self.model_1_button.clicked.connect(lambda: self.get_model('rubert'))
        self.model_2_button = QPushButton("Загрузить Universal sentence encoder")
        self.model_2_button.clicked.connect(lambda: self.get_model('use'))
        self.model_choose_layout.addWidget(self.model_1_button)
        self.model_choose_layout.addWidget(self.model_2_button)

        
        self.data_choose_layout = QHBoxLayout() # второй layout
        self.get_data_gf_button = QPushButton("Получить данные из Google forms")
        self.get_data_gf_button.clicked.connect(lambda: self.get_data('gf'))
        self.get_data_csv_button = QPushButton("Получить данные из csv")
        self.get_data_csv_button.clicked.connect(lambda: self.get_data('csv'))
        self.data_choose_layout.addWidget(self.get_data_gf_button)
        self.data_choose_layout.addWidget(self.get_data_csv_button)
        
        
        self.compute_button = QPushButton("Запустить модель")
        self.compute_button.clicked.connect(self.start_process)
        self.data_status = QLabel("Статус: данные не загружены")
        self.model_status = QLabel("Статус: модель не загружена")
        frame1_1_layout.addLayout(self.app_descr_layout)
        frame1_2_layout.addWidget(QLabel("Выберите модель для создания эмбеддингов",
                                         alignment=QtCore.Qt.AlignCenter))
        frame1_2_layout.addWidget(self.model_status)
        frame1_2_layout.addLayout(self.model_choose_layout)
        frame1_2_layout.addWidget(QLabel("Выберите файл для извлечения данных",
                                         alignment=QtCore.Qt.AlignCenter))
        frame1_2_layout.addWidget(self.data_status)
        frame1_2_layout.addLayout(self.data_choose_layout)
        
        frame1_2_layout.addWidget(self.compute_button)
        embeddings = np.load("dima/эмбеддинги2d.npy")
        clusters = np.load("dima/метки_кластеров.npy")
        answers = np.load("dima/ответы_сотрудников.npy")
        self.window = BarChartWidget()
        # window.apply(embeddings, answers, clusters)
        # window.initUI()
        frame2_layout.addWidget(self.window)


    def get_data(self, type_of_data):
        if type_of_data == 'csv':
            csv_file, _ = QFileDialog.getOpenFileName(self, "Выберите директорию")
            self.data = pd.read_csv(csv_file).iloc[1:,0].reset_index(drop=True)
            self.data_status.setText("Выбранный файл: {}".format(csv_file) )
        elif type_of_data == 'gf':
            pass
    def get_model(self, model_name):
        if model_name == 'use':
            self.embedding_model = universal_sentence_encoder()
            self.model_status.setText('Загружена модель USE (universal sentence encoder)')
        elif model_name == 'rubert':
            self.embedding_model = RuBertEmbedder()
            self.model_status.setText('Загружена модель RuBert')
    def start_process(self):
        embeddings = self.embedding_model.transform(self.data)
        embeddings = np.array(embeddings)
        clusterer = Clasterer(method = 'svd', n_components=2, max_clusters=8)
        embeddings_reduced, cluster_labels = clusterer.transform(embeddings)
        self.window.apply(embeddings = embeddings_reduced,
                          answers = np.array(self.data, dtype = np.str_),
                          clusters = cluster_labels)
        self.window.initUI()
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = UI()
    ex.show()
    sys.exit(app.exec_())
        
        
        
        












