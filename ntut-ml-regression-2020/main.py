from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
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

def build_model():
    model = keras.Sequential([keras.layers.Dense(64,input_dim=x_train_normalize.shape[1],kernel_initializer='normal',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    keras.layers.Dense(128,kernel_initializer='normal',activation='relu'),
    keras.layers.Dense(128,kernel_initializer='normal',activation='relu'),
    keras.layers.Dense(64,kernel_initializer='normal',activation='relu'),
    keras.layers.Dense(1,kernel_initializer='normal')
  ])

    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    model.compile(loss='mae',
        optimizer=optimizer,
        metrics=['mae'])
    return model

if __name__ == '__main__':
   
    x_train, y_train, x_valid, y_valid, x_test = load_data_origin()
    x_train_normalize, x_valid_normalize, x_test_normalize = normalize_origin(x_train)
 
print("please input EPOCHS: ")   
a= input()
EPOCHS=int(a)

model = build_model()
model.summary()
callbackcp = ModelCheckpoint("t.h5", monitor='val_mae', verbose=0, save_best_only=True,
                            save_weights_only=False, mode='min', period=1)
history = model.fit(x_train_normalize, y_train,batch_size =512, epochs=EPOCHS,
                      validation_data=(x_valid_normalize, y_valid), verbose=1,
                      callbacks=[callbackcp])
model.save('a.h5')

"""Plot loss v.s. epochs"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='best')
index = history.history['val_loss'].index(min(history.history['val_loss']))
print("\nMininum validation loss: epochs = ", index )
print("Set num_epoch = ", index+1)
plt.show()
print('===>plot OK')

[loss, mae] = model.evaluate(x_valid_normalize, y_valid, batch_size= 16,verbose=0)

print("Testing set Mean Abs Error: ${:8.2f}".format(mae))  

model = tf.keras.models.load_model('a.h5')

pred=model.predict(x_test_normalize)

a=np.arange(1,6486)
sub = pd.DataFrame()
sub['id'] = a
sub['price'] = pred
sub.to_csv('sub.csv',index=False)
print('===>sub.csv OK')