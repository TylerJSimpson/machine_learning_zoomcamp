#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '0')


# In[2]:


get_ipython().system('wget https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/chapter7-model/xception_v4_large_08_0.894.h5 -O clothing-model.h5')


# In[1]:


get_ipython().system('python -V')


# In[2]:


import numpy as np

import tensorflow as tf
from tensorflow import keras

tf.__version__


# In[3]:


get_ipython().system('wget http://bit.ly/mlbookcamp-pants -O pants.jpg')


# In[4]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[5]:


model = keras.models.load_model('clothing-model.h5')


# In[6]:


img = load_img('pants.jpg', target_size=(299, 299))

x = np.array(img)
X = np.array([x])

X = preprocess_input(X)


# In[7]:


preds = model.predict(X)


# In[8]:


preds


# In[9]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]


# In[10]:


dict(zip(classes, preds[0]))


# ## Convert Keras to TF-Lite

# In[20]:


converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('clothing-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[21]:


get_ipython().system('ls -lh')


# In[22]:


import tensorflow.lite as tflite


# In[23]:


interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[30]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[34]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


# ## Removing TF dependency

# In[36]:


from PIL import Image


# In[38]:


with Image.open('pants.jpg') as img:
    img = img.resize((299, 299), Image.NEAREST)


# In[45]:


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


# In[47]:


x = np.array(img, dtype='float32')
X = np.array([x])

X = preprocess_input(X)


# In[49]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[50]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


# ## Simpler way of doing it

# In[52]:


get_ipython().system('pip install keras-image-helper')


# In[7]:


get_ipython().system('pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime')


# In[3]:


#import tensorflow.lite as tflite
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


# In[4]:


interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# In[5]:


preprocessor = create_preprocessor('xception', target_size=(299, 299))


# In[6]:


url = 'http://bit.ly/mlbookcamp-pants'
X = preprocessor.from_url(url)


# In[7]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)


# In[8]:


classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

dict(zip(classes, preds[0]))


# In[ ]:




