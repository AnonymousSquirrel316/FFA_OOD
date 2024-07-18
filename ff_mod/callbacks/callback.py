from abc import ABC, abstractmethod

class Callback:
    
    def __init__(self) -> None:
        self.epoch = 0
        self.train_step = 0
        self.test_step = 0
        pass
    
    def reset(self):
        self.epoch = 0
        self.train_step = 0
        self.test_step = 0
        pass
    
    def next_epoch(self):
        self.epoch += 1
        self.train_step = 0
        self.test_step = 0
        pass
    
    def on_train_batch_start(self, *args, **kwargs):
        pass
    
    def on_train_batch_end(self, *args, **kwargs):
        self.train_step += 1
        pass
    
    def on_test_batch_start(self, *args, **kwargs):
        pass
    
    def on_test_batch_end(self, *args, **kwargs):
        self.test_step += 1
        pass
    
    def on_train_epoch_end(self, *args, **kwargs):
        pass
    
    def on_train_epoch_start(self, *args, **kwargs):
        pass
    
    def on_test_epoch_end(self, *args, **kwargs):
        pass
    
    def on_test_epoch_start(self, *args, **kwargs):
        pass



class CallbackList:
    
    def __init__(self) -> None:
        self.callbacks = []
        pass
    
    def add(self, callback : Callback):
        self.callbacks += [callback]
    
    def reset(self):
        for callback in self.callbacks:
            callback.reset()
    
    def next_epoch(self):
        for callback in self.callbacks:
            callback.next_epoch()
    
    def on_train_batch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_batch_start(*args, **kwargs)
    
    def on_train_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_batch_end(*args, **kwargs)
    
    def on_test_batch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_test_batch_start(*args, **kwargs)
    
    def on_test_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_test_batch_end(*args, **kwargs)
    
    def on_train_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_epoch_end(*args, **kwargs)
    
    def on_train_epoch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_epoch_start(*args, **kwargs)
    
    def on_test_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_test_epoch_end(*args, **kwargs)
    
    def on_test_epoch_start(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_test_epoch_start(*args, **kwargs)
    