from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

def load_data_origin():
    
    print('===>load_data OK ')
    
    """prepare training data"""
    train_data = pd.read_csv('train-v3.csv')
    x_train_data = train_data.drop(['id','price'],axis=1)
    x_train = x_train_data.values
    y_train = train_data['price'].values

    """prepare validation data"""
    valid_data = pd.read_csv('valid-v3.csv')
    x_valid_data = valid_data.drop(['id','price'],axis=1)
    x_valid = x_valid_data.values
    y_valid = valid_data['price'].values

    """prepare testing data"""
    test_data = pd.read_csv('test-v3.csv')
    x_test_data = test_data.drop(['id'],axis=1)
    x_test = x_test_data.values
    return x_train, y_train, x_valid, y_valid, x_test


def normalize_origin(x_train):
    
    print('===>normalizing OK')
    
    """normalize training data"""
    tmp = x_train
    mean, std = tmp.mean(axis=0),tmp.std(axis=0)
    x_train_normalize = (x_train - mean) / std                       
    x_valid_normalize = (x_valid - mean) / std
    x_test_normalize = (x_test - mean) / std
    return x_train_normalize, x_valid_normalize, x_test_normalize




if __name__ == '__main__':
   
    x_train, y_train, x_valid, y_valid, x_test = load_data_origin()
    x_train_normalize, x_valid_normalize, x_test_normalize = normalize_origin(x_train)
    
    model = tf.keras.models.load_model('t.h5')

[loss, mae] = model.evaluate(x_valid_normalize, y_valid, batch_size= 8,verbose=0)

print("Testing set Mean Abs Error: ${:8.2f}".format(mae))                   

pred=model.predict(x_test_normalize)

a=np.arange(1,6486)
sub = pd.DataFrame()
sub['id'] = a
sub['price'] = pred
sub.to_csv('sub.csv',index=False)
print('===>sub.csv OK')
