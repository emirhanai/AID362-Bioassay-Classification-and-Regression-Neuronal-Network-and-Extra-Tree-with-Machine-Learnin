import numpy as np
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.model_selection import *
import pandas as pd

df = pd.read_csv('bio_chemist_data.csv')


def numerical_class(i):
    if i == 'Inactive':
        return 0
    else:
        return 1


df['label'] = df['Outcome'].apply(numerical_class)

X = df.drop(['Outcome','label'],axis='columns')
XX = X.iloc[:4000,:].values
y = df[['label']]
yy = y.iloc[:4000,:].values

print(X.shape)
print(y.shape)

#0.146
#0.083
for i in np.arange(0,1,1):
    X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.16,random_state=7,shuffle=True,stratify=None)
#18

    from sklearn.neural_network import *

    model_ml_emir = ExtraTreesClassifier(n_estimators=23,criterion="gini",max_features="auto",random_state=131)

    #model_ml_emir = MLPClassifier(activation="relu",
                                  #solver="adam",
                                  #batch_size=200,
                                  #hidden_layer_sizes=(100,),random_state=17,
                                  #learning_rate='constant',
                                  #alpha=0.0006,
                                  #beta_1 = 0.9,
                                  #beta_2=0.4)

    model_ml_emir.fit(X_train, y_train.ravel())

    prediction = model_ml_emir.predict(X_test)

    accuracy_score(y_pred=prediction, y_true=y_test)

    print("X",i)

    print("Machine Learning Software is the Accuracy Score: {0} "
          .format(accuracy_score(y_pred=prediction, y_true=y_test)))
    print("Machine Learning Software is the Precision Score: {0} "
          .format(precision_score(y_pred=prediction, y_true=y_test)))
    print("Machine Learning Software is the Recall Score: {0} "
          .format(recall_score(y_pred=prediction, y_true=y_test)))
    print("Machine Learning Software is the F1 Score: {0} "
          .format(f1_score(y_pred=prediction, y_true=y_test)))

    import matplotlib.pyplot as plt

    feature_import = model_ml_emir.feature_importances_

    a = np.std([h.feature_importances_ for h in
                model_ml_emir.estimators_],
               axis=0)

    df_x = pd.DataFrame(X_test,columns=X.columns)
    df_y = pd.DataFrame(y_test,columns=y.columns)
    #print(df_y)

    #print(len(prediction))
    #print(len(X_test))


    #plt.scatter(X_test,y_test)
    #plt.plot(prediction,df_x,color = "red")
    #plt.xlabel('Feature Labels')
    #plt.ylabel('Feature Importances')
    #plt.title('Comparison of different Feature Importances')
    #plt.show()

    import pylab as pl

    from sklearn.decomposition import PCA

    model_ozone = PCA(n_components=2).fit(X_train)
    model_ozone_2d = model_ozone.transform(X_train)

    for i in range(0, model_ozone_2d.shape[0]):
        if y_train[i] == 0:
            c1 = pl.scatter(model_ozone_2d[i, 0], model_ozone_2d[i, 1], color='r', edgecolors='y', marker='*',
                            linewidths=1)

        elif y_train[i] == 1:
            c2 = pl.scatter(model_ozone_2d[i, 0], model_ozone_2d[i, 1], color='g', edgecolors='y', marker='o',
                            linewidths=1)
    import matplotlib.pyplot as plt

    pl.legend([c1, c2], ['Inactive', 'Active'])
    plt.title('Bio Inactive/Active Classification')
    #pl.show()

    import plotly.express as px

    #print(X.shape)

    #model creating of regression in Extra Tree Regressor of prediction

    model_emir_regress_predict = ExtraTreesRegressor(criterion="mse",max_features="auto",
                                                     n_jobs=-1,n_estimators=1)
    #model_emir_regress_predict = MLPRegressor(hidden_layer_sizes=(200,),activation="relu",
                                              #solver="adam",batch_size="auto")

    model_emir_regress_predict.fit(X_train,y_train)

    predict_regress = model_emir_regress_predict.predict(X_test)

    #print(r2_score(y_test,predict_regress))

    print("Accuracy: ",r2_score(y_test,predict_regress))
    print("CM: ",confusion_matrix(y_test,predict_regress))

    while True:
        predict = model_emir_regress_predict.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, 0, -1.1569, 1.1837, -1.9082, 2.0213, -2.7385, 2.9148, -3.5948,
                                                       3.8259, -0.7602, 1.5808, -1.5435, 2.4587, -2.3527, 3.3599,
                                                       -3.1856, 4.2714, -1.0911, 1.1333, -1.9381, 1.9813, -2.8029,
                                                       2.8675, -3.6753, 3.773, 2.704, 119.85, 4, 6, 0, 424.569, 0, 0]])
        predict_to_np = np.array(predict)
        np_to_list = predict_to_np.tolist()

        if np_to_list == [0]:
            print("Prediction of Bio Lab Result: Inactive")
            break
        elif np_to_list == [1]:
            print("Prediction of Bio Lab Result: Active")
            break


    fig = px.sunburst(df, path=['MW', 'BBB'],
                      values='label',
                      color_discrete_map={'(?)':'black', 0:'gold', 1:'darkblue'})
    fig.show()