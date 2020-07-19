#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
model=pickle.load(open('house_price.pkl','rb'))


# In[2]:


from flask import Flask,url_for,render_template,request
import numpy as np


# In[3]:


app=Flask(__name__)


# In[ ]:


@app.route('/')
def home():
    return render_template('home.html')
  
@app.route('/predict',methods=['POST'])
def predict():
    #if request.method=='POST':
    inp=[i for i in request.form.values()]
    
    arr=np.array(inp)
    y=arr.astype(np.float)
    y1=[y]
    pred=model.predict(y1)[0]
    output=int(pred)
    
    return render_template('home.html',predicted="Approximate Price will be : {}".format(output)  )
    
if __name__=='__main__':
    app.run()


# In[1]:





# In[ ]:




