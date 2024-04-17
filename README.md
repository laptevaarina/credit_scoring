# credit_scoring
За основу было взято [baseline решение для RNN](https://github.com/SmirnovValeriy/dl-fintech-bki/tree/master/rnn_baseline).
Лучший результат показало решение BiLSTM (BiLSTM_submission.csv):
- без регуляризации
- с конкатенацией входа и выхода LSTM
- с постоянным learning rate = 1e-4
- с AdaptiveMaxPooling по схеме на картинке:
![image](https://github.com/laptevaarina/credit_scoring/assets/45456174/42cc1645-dcc7-4140-9d1d-f267b59b477a)
**Score:** 0.7663312161
