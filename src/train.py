
from sklearn.linear_model import LinearRegression
from data_preprocessing import (
    Data_preprocessing,
    DataRead,
    RemoveColumn,
    SaparateInputOutputFeatureIntoXY,
    LinearRegressionpredictor)
from sklearn.metrics import mean_absolute_error, r2_score
import joblib



class studentMarksTrainer():

    def __init__(self,file_path):
        self.file_path = file_path

        self.reader = None
        self.cleaner = None
        self.splitter =None
        self.processor = None

        self.model = None
        self.predictor = None

        self.x = None
        self.y = None
        self.x_train_scale = None
        self.x_test_scale =None
        self.y_train = None
        self.y_test = None
        self.predictions = None

    # read data 
    def load_data(self):
        self.reader = DataRead(self.file_path)

    #Remove unnecessary column
    def clean_data(self):
        self.cleaner = RemoveColumn(
            self.reader.df,
            ["student_id"]
        )

    # Separate x & y
    def prepare_feature(self):
        self.splitter = SaparateInputOutputFeatureIntoXY(
            self.cleaner.df,
            target_column="final_marks"
        )
        self.x, self.y = self.splitter.Splitfeatures()

    #split and scale data
    def preprocess_data(self):
        self.processor = Data_preprocessing(
            self.x,
            self.y 
        )
        self.processor.split_data()

        (
            self.x_train_scale,
            self.x_test_scale,
            self.y_train,
            self.y_test,
        ) = self.processor.scale_feature()

    # create and train model
    def train_model(self):
        self.model = LinearRegression()

        self.model.fit(
            self.x_train_scale,
            self.y_train
        )

    #make predictions
    def make_preidction(self):
        self.predictor = LinearRegressionpredictor(
            self.model 
        )
        self.predictions = self.predictor.prediction(
            self.x_test_scale
        )

    # evaluate model accuracy
    def evaluate_model(self):
        mae = mean_absolute_error(
            self.y_test,
            self.predictions   
        )

        r2 = r2_score(
            self.y_test,
            self.predictions
        )

        print("MAE0",mae)
        print("r2 Score", r2)

    #save model through Joblib librabry
    def save_model(self):
        joblib.dump(
            self.model,
            r"C:\Users\Shubham\Desktop\students_marks_prediction\models\model.pkl"
        )

        print("Model Save Successfully !")


    def save_scaler(self):
        joblib.dump(
            self.processor.scaler,
            r"C:\Users\Shubham\Desktop\students_marks_prediction\models\scaler.pkl"
        )
        print("Scaler saved successfully!")


    # Run complete pipline
    def run(self):

        self.load_data()
        self.clean_data()
        self.prepare_feature()
        self.preprocess_data()
        self.train_model()
        self.make_preidction()
        self.evaluate_model()
        self.save_model()
        self.save_scaler()

        print(
            "First Five Prediction:", self.predictions[:5].round().astype(int)
        )


#create object
trainer = studentMarksTrainer(
    "C:\\Users\\Shubham\\Desktop\\students_marks_prediction\\Data\\students.csv"
)

#run complete ML pipline

trainer.run()



