import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataRead():

    def __init__(self, file_path="C:\\Users\\Shubham\\Desktop\\students_marks_prediction\\Data\\students.csv"):

        self.df = pd.read_csv(file_path)

        print(f"Data Load Successfully in DataFrame {self.df}")


class RemoveColumn():

# droping the column which is not needed
    def __init__(self, df, column_to_drop):

        
        self.df = df

    #Modify self.df using a column drop operation
        self.column_to_drop = column_to_drop
        self.drop_columns()

    def drop_columns(self):
    #drop column safely directly from self.df
        self.df = self.df.drop(self.column_to_drop, axis=1 ,errors="ignore")

        print(f"Droped colums: {self.column_to_drop}")


class SaparateInputOutputFeatureIntoXY():

    def __init__(self, df, target_column):

        self.df = df

        self.target_column = target_column

    #Attribute to hold split features and target feature

        self.x = None
        self.y = None

    def Splitfeatures(self):
            
    #now Saparate the indipendance features and dependent features x,y
            
        self.x = self.df.drop(columns={self.target_column})

        self.y = self.df[self.target_column]

        print(f"X feature shape: {self.x.shape}")
        print(f"y feature shape: {self.y.shape}")

        return self.x , self.y
    

class Data_preprocessing():

    def __init__(self, x, y, test_size=0.2, random_state=42):
        self.x= x
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

    # now they are the placeholder of actual data
        self.x_train = None
        self.x_test = None
        self.y_train = None 
        self.y_test = None    

    # initializing standarscaler instance here

        self.scaler = StandardScaler()
        self.x_trained_scale = None
        self.x_test_scale = None


    def split_data(self):

    # now split data into train test using model_selection import train_test_split

        self.x_train, self.x_test, self.y_train, self.y_test = (
            train_test_split(

                self.x,
                self.y,
                test_size= self.test_size,
                random_state= self.random_state
            )
        )

        print(f"Training feature shape: {self.x_train.shape}")
        print(f"Test feature shape: {self.x_test.shape}")

    def scale_feature(self):

    # fit anad transform traing set
        self.x_trained_scale = self.scaler.fit_transform(self.x_train)
    # transform testing set 
        self.x_test_scale = self.scaler.transform(self.x_test)

        return (
            self.x_trained_scale,
            self.x_test_scale,
            self.y_train,
            self.y_test
        )



class LinearRegressionpredictor():

    def __init__(self , lr_model , feature_names=None):

    #stores the trained linearRegression model and feature_names

        self.lr_model = lr_model
        self.feature_names = feature_names

    def prediction(self, input_data ):
    # formats or convert input data into pandas df if data passed as dict , list and returns prediction
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data 
    # allways call and return model  prediction
            return self.lr_model.predict(input_df)


print("successfully made class")

    






