# Hidden-size sensitivity: recurrent LSTM(A,F) with LSTM hidden size 16 (default is 32).
# Everything else identical to config_lateral_lstm_af.py. Tests whether the ~180-day effective
# memory and the F_add magnitude depend on encoder capacity (dropout held at 0.3).
from config_lateral_lstm_af import *

LSTM_HIDDEN_SIZE = 16
