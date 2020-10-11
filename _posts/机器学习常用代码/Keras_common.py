### earlystop
es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=10,restore_best_weights=True)
model.fit(X_train2,y_train2, batch_size = 1024,
            validation_data  = (X_test2,y_test2), 
            epochs=300,verbose=1, callbacks=[es,lr])

### lr_decay
def lr_decay(index_):
    if index_ < 15:
        return 0.0005
    elif  index_ < 30:
        if  index_ % 2 ==0:
            return 0.0001
        else:
            return 0.0003                  
    elif  index_ < 40:
        if  index_ % 2 ==0:
            return 0.0001
        else:
            return 0.0002          
    else:
        return 0.0002
lr = keras.callbacks.LearningRateScheduler(lr_decay)
model.fit(X_train2,y_train2, batch_size = 1024,
            validation_data  = (X_test2,y_test2), 
            epochs=300,verbose=1, callbacks=[es,lr])


### ModelCheckpoint
### 和ES 不重复 他会跑完所有epoch 但是不早停
from keras.callbacks import ModelCheckpoint
### default parameter
ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
model_checkpoint = ModelCheckpoint("model_" + str(fold) + ".hdf5",
                                    save_best_only=True, verbose=1, monitor='val_loss', mode='auto')

### ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
###factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
###patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
### mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。