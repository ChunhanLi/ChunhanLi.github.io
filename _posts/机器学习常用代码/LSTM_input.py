### Stateful 情况下
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))

### 普通情况下
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))