""""
LSTM 셀을 기본 단위로하는 Seq2Seq 클래스가 정의된 파일입니다.
LSTM을 이해하기 전에 우선 순환신경망 RNN (Recurrent Neural Network)에 대한 이해가 필요합니다.


Author : Hyunwoong
When : 7/31/2019
Homepage : github.com/gusdnd852
"""



# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys
class Seq2Seq:

    vector_size = 256
    data =