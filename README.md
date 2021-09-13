# **AID362 Bioassay Classification and Regression (Neuronal Network and Extra Tree) with Machine Learning**
I developed Machine Learning Software with multiple models that predict and classify AID362 biology lab data. Accuracy values are 99% and above, and F1, Recall and Precision scores are average (average of 3) 78.33%. The purpose of this study is to prove that we can establish an artificial intelligence (machine learning) system in health. With my regression model, you can predict whether it is Inactive or Inactive (Neural Network or Extra Trees). In classification (Neural Network or Extra Trees), you can easily classify the provided data whether it is Inactive or Active.

_Example:_ 

    `###Regressor Model

    model_emir_regress_predict = ExtraTreesRegressor(criterion="mse",max_features="auto",
                                                     n_jobs=-1,n_estimators=1)

    model_emir_regress_predict = MLPRegressor(hidden_layer_sizes=(200,),activation="relu",
                                              #solver="adam",batch_size="auto")`

    ###Classifier Model

    `model_ml_emir = ExtraTreesClassifier(n_estimators=23,criterion="gini",max_features="auto",random_state=131)

    model_ml_emir = MLPClassifier(activation="relu",
                                  #solver="adam",
                                  #batch_size=200,
                                  #hidden_layer_sizes=(100,),random_state=17,
                                  #learning_rate='constant',
                                  #alpha=0.0006,
                                  #beta_1 = 0.9,
                                  #beta_2=0.4)`
 
**I am happy to present this software to you!**

###**The coding language used:**

`Python 3.9.6`

###**Libraries Used:**

`Sklearn`

`Pandas`

`Numpy`

`Matplotlib`

`Pylab`

`Plotly`

### **Tags**

_business, earth and nature, health, biology, chemistry, biotechnology, Machine Learning, Python, Artificial Intelligence, Neural Networks, Extra Tree Classifier, Extra Tree Regressor, Software_


### **Developer Information:**

Name-Surname: **Emirhan BULUT**

Contact (Email) : **emirhan.bulut@turkiyeyapayzeka.com**

LinkedIn : **[https://www.linkedin.com/in/artificialintelligencebulut/][LinkedinAccount]**

Data Source: [DataSource]

[LinkedinAccount]: https://www.linkedin.com/in/artificialintelligencebulut/

Official Website: **[https://www.emirhanbulut.com.tr][OfficialWebSite]**

[OfficialWebSite]: https://www.emirhanbulut.com.tr

[DataSource]: https://kaggle.com



<img src="https://raw.githubusercontent.com/emirhanai/AID362-Bioassay-Classification-and-Regression-Neuronal-Network-and-Extra-Tree-with-Machine-Learnin/main/bio-machine-learning-emirhan-project.jpg" alt="bio-machine-learning-emirhan-project.jpg">

<img src="https://raw.githubusercontent.com/emirhanai/AID362-Bioassay-Classification-and-Regression-Neuronal-Network-and-Extra-Tree-with-Machine-Learnin/main/bio_inactive-active_plot.png" alt="bio_inactive-active_plot.png">

<img src="https://raw.githubusercontent.com/emirhanai/AID362-Bioassay-Classification-and-Regression-Neuronal-Network-and-Extra-Tree-with-Machine-Learnin/main/bio_plot.png" alt="bio_plot.png">
