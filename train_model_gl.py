from training_model_gru import main as gru_main
from training_model_lstm import main as lstm_main

gru_main(50)
gru_main(100)
gru_main() 

lstm_main(50)
lstm_main(100)
lstm_main()