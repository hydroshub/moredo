# Hidden-size sensitivity: recurrent LSTM(A,F) with LSTM hidden size 48 (default is 32).
# Everything else identical to config_lateral_lstm_af.py. Fills the gap between the 32- and
# 64-unit runs to locate where the F_add magnitude/memory start to change (dropout held at 0.3).
from config_lateral_lstm_af import *

LSTM_HIDDEN_SIZE = 48
