from src.data_generation import read_dataset, read_weather_dataset, preprocess_dataset, denormalize_label
from src.model import DC_CNN_Model, DC_CNN_LSTM_Model, lstm_model
from src.metric import evaluate, evaluate_all
import itertools
import time
import os
import tensorflow as tf
import numpy as np
import pandas as pd



def run_model(data_name, weather_name, result_file, future_target_in, past_history_in, batch_size_in, epochs_in, params, model, metric_in, buffer_size=10000, validation_size=0.2, seed=1) :
   
    checkpoint_path = "training/cp.weights.h5"

    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    ## Read dataset
    data = read_dataset(data_name)
    weather = read_weather_dataset(weather_name)
    ## Write result file
    current_index = 0
    if current_index == 0 :
      with open(result_file, 'w') as resfile :
        resfile.write(';'.join([str(a) for a in ['MODEL', 'MODEL_DESCRIPTION', 'FORECAST_HORIZON', 'PAST_HISTORY', 'BATCH_SIZE', 'EPOCHS'] + metric_in + ['val_' + m for m in metric_in] + ['loss', 'val_loss']]) + "\n")
    
    for past_history, batch_size, epochs in list(itertools.product(past_history_in, batch_size_in, epochs_in)) :  
      ## Preprocess time series dataset and get x, y for train, test and validation
      x_train, y_train, x_val, y_val, x_test, y_test, norm_params = preprocess_dataset(data, weather, past_history, future_target_in)
    
      train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(buffer_size).batch(batch_size).repeat()
      val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size).repeat()
      test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
      ## Create models
      model_list = {}

      if model == 'lstm' :
          model_list = {'lstm_Model_{}'.format(j) : (lstm_model, [x_train.shape[-2:], future_target_in, *params]) for j, params in enumerate(itertools.product(*params.values()))}
      
      if model == 'dccnn' :
          model_list = {'DCCNN_Model_{}'.format(j) : (DC_CNN_Model, [x_train.shape[-2:], future_target_in, *params]) for j, params in enumerate(itertools.product(*params.values())) if params[1] * params[2] * params[3][-1] >= past_history}
           
      if model == 'dclstm' : 
          model_list = {'DC_CNN_LSTM_Model{}'.format(j) : (DC_CNN_LSTM_Model, [x_train.shape[-2:], future_target_in, *params]) for j, params in enumerate(itertools.product(*params.values())) if params[1] * params[2] * params[3][-1] >= past_history}
      
      steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
      validation_steps = steps_per_epoch if val_data else None
      
      for model_name, (model_function, params) in model_list.items() :
        model = model_function(*params)
        print(model.summary())
        model.compile(loss='mae', optimizer='adam', metrics=['mse']) 
        print(*params)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, monitor='val_loss', save_best_only=True)
        
        ## Train model
        history = model.fit(train_data, epochs=epochs, validation_data=val_data, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, callbacks=[cp_callback])
        model.load_weights(checkpoint_path)
        model.save('model/'+ model_name + '.h5')
        val_metrics = {}
    
        


        ## Get validation result metrics
        if validation_size > 0 :
          val_forecast = model.predict(x_val)
          val_forecast = denormalize_label(val_forecast, norm_params)
          y_val_denorm = denormalize_label(y_val, norm_params)
          x_val_denorm = denormalize_label(x_val[:, :, -1], norm_params)    
            
          val_metrics = evaluate(y_val_denorm, val_forecast, metric_in)
          print('Val metrics : ', val_metrics)
        
        ## Get test result metrics
        test_forecast = model.predict(test_data)
        test_forecast = denormalize_label(test_forecast, norm_params)
        y_test_denorm = denormalize_label(y_test, norm_params)
        x_test_denorm = denormalize_label(x_test[:, :, -1], norm_params)
        print("test_forecast" , test_forecast)
        # Get the current hour from the last timestep in x_test
        # Assumes hour is the second-last feature in x_test
        current_hours = x_test[:, -1, -2].astype(int)  # shape: (n_samples,)

        # Compute forecasted hours for the next 24 hours
        forecast_hours = (current_hours[:, None] + np.arange(test_forecast.shape[1])) % 24  # shape: (n_samples, 24)

        # Create mask: 1 if daylight (6 to 18), else 0
        daylight_mask = ((forecast_hours >= 6) & (forecast_hours <= 18)).astype(int)

        # Apply daylight mask to predictions
        test_forecast = test_forecast * daylight_mask


        test_metrics = evaluate(y_test_denorm, test_forecast, metric_in)
        print('Test metrics : ', test_metrics)
    
        val_metrics = {'val_' + k: val_metrics[k] for k in val_metrics}
        print('Val metrics : ', val_metrics)
        # Ensure no negative values
        test_forecast = np.clip(test_forecast, 0, None)

        timestamps = pd.date_range(start='2020-01-01', periods=test_forecast.shape[0], freq='H')  # Replace this with actual timestamps from your data

          # Create DataFrame
        df = pd.DataFrame(test_forecast, columns=[f'hour_{i+1}' for i in range(test_forecast.shape[1])])
        df.insert(0, "timestamp", timestamps)

          # Save to CSV
        df.to_csv('predicted_24hr_output_with_timestamps.csv', index=False)
        print("All 24-hour predictions saved with timestamps to 'predicted_24hr_output_with_timestamps.csv'")


        # Print first prediction (next 24 hours)
        print("Sample 24-hour prediction:")
        print(test_forecast[0])  # first row, 24 hours

        # Optional: Save all predictions to CSV
       
        df = pd.DataFrame(test_forecast, columns=[f'hour_{i+1}' for i in range(test_forecast.shape[1])])
        df.to_csv('predicted_24hr_output.csv', index=False)
        print("All 24-hour predictions saved to 'predicted_24hr_output.csv'")

        
        test_forecast = test_forecast[:7]
        df_7day = pd.DataFrame(test_forecast, columns=[f"hour_{i+1}" for i in range(test_forecast.shape[1])])
        df_7day.to_csv("predicted_7day_output.csv", index=False)
        ## Save a result
        model_metric = {'MODEL' : model_name,
                        'MODEL_DESCRIPTION' : params,
                        'FORECAST_HORIZON' : future_target_in,
                        'PAST_HISTORY' : past_history,
                        'BATCH_SIZE' : batch_size,
                        'EPOCHS' : epochs,
                        **test_metrics,
                        **val_metrics,
                        'loss' : history.history['loss'],
                        'var_loss' : history.history['val_loss'],    
                        }
        

        ## Write a result file


        with open(result_file, 'a') as resfile :
          resfile.write(';'.join([str(a) for a in model_metric.values()]) + "\n")
      print("model_metric------------------",model_metric)

    
  