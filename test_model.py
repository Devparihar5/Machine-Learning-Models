#importing required Modules
from Modules import *
warnings.filterwarnings("ignore")#remove any kind of warnings
path = 'models/'


#student marks prediction
def Marks():
    hrs = int(input("Enter Study time: "))
    load_model = pickle.load(open(path + 'student_marks_model.pkl','rb'))
    data = [[hrs]]
    result = load_model.predict(data)
    return round(result[0],2)


#iris species prediction
def species():
    #geting data from user
    sl=float("Enter Sepal Length:")
    sw=float("Enter Sepal Width:")
    pl=float("Enter Petal Length:")
    pw=float("Enter Petal Width:")
    load_model = pickle.load(open(path + 'iris_model.pkl','rb'))
    #data = pd.DataFrame([[sl, sw, pl, pw]])
    data = [[sl, sw, pl, pw]]
    predict_spe = load_model.predict(data)
    if predict_spe == 0:
        return "Species: Iris-setosa"
    elif predict_spe == 1:
        return "Species: Iris-versicolor"
    elif predict_spe == 2:
        return "Species: Iris-verginica"

# predict cancer
def cancer():
    load_model = pickle.load(open(path + 'cancer_prediction_model.pkl','rb'))
    input_data1 = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean',
                   'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se',
                   'concave_points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
                   'compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst']
    ninput_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
    print("In Which Data you want to test this model:")
    print("'Custom'  or 'Pre-defiend' data")
    print("Enter 'C' for custom and 'P' for Pre-defiend data")
    option=input("Enter your choice: ")
    option=option.upper()
    if option == 'P':
        input_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
        for i in range(len(input_data1)):
                                   item1=input_data1[i]
                                   item=input_data[i]

                                   print(item1 + ' = ' + str(item))
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = load_model.predict(input_data_reshaped)
        #model Prediction
        print('\n\n')
        if(prediction[0]==0):
            print('The Breast Cancer type is : Malignant')
        else:
            print('The Breast Cancer type is : Benign')
    elif option == 'C':
        lis = []
        for i in range(len(input_data1)):
        
            name=input_data1[i]
            name = float(input(f"Enter the value of {name}: ",))
            lis.append(name)
        # changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(lis)
        # reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
                
        prediction = load_model.predict(input_data_reshaped)
        #model Prediction
        print('\n\n')
        if(prediction[0]==0):
            print('The Breast Cancer type is : Malignant')
        else:
            print('The Breast Cancer type is : Benign')
    else:
        print("Please Enter Correct option")
    
#spam mail predictio
def spamham():
    load_model = pickle.load(open(path + 'spam_mail_prediction_model.pkl','rb'))
    loaded_vectorizer = pickle.load(open('feature_extraction_vect.pkl', 'rb'))
    input_mail = input('Enter mail-->')
    input_mail = [input_mail] 
    #input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times."]
    # convert text to feature vectors
    input_mail_features = loaded_vectorizer.transform(input_mail)
    #making prediction
    prediction = load_model.predict(input_mail_features)
    if (prediction[0]==1):
        print('Entered mail Type is : HAM MAIL')
    else:
        print('Entered mail Type is : SPAM MAIL')

#titanic survival prediction
def survival():
    load_model = pickle.load(open(path + 'Titanic_survival_prediction_model.pkl','rb'))
    #getting data from user
    pclassNo=int(input("Enter Person's pclass number: "))
    print("For Feamle=0 and For Male=1")
    gender=int(input("Enter the gender of Person's: "))
    age=int(input("Enter the age of person's:"))
    fare=float(input("Enter the person's fare: "))
    #fit all data into person dataFrame
    person=[[pclassNo,gender,age,fare]]
    result=load_model.predict(person)
    print('\n')
    if result==1:
        print("Person Might be survived")
    else:
        print("Person Might not be Survived ")

#employee salary prediction
def salary():
    load_model = pickle.load(open(path + 'Employee_salary_prediction_model.pkl','rb'))
    exp=int(input("Enter year of experience: "))
    #fit all data into employe dataFrame
    employe = pd.DataFrame([exp])
    predictions = load_model.predict(employe)
    print(f'Expected Salary :{int(predictions[0])} $')

#customer next purchase prediction
def purchase():
    load_model = pickle.load(open(path + 'next_purchase_prediction_model.pkl','rb'))
    age=int(input("Enter the age of customer-->"))
    salary=int(input("Enter the salary of customer-->"))
    #fit all data into newCust dataFrame
    newCust = [[age,salary]]
    result = load_model.predict(newCust)
    print('\n')        
    if result==1:
        print("Response: Customer will buy your product")
    elif result==0:
        print("Response: Customer won't buy your product!!")

#house price prediction
def house():
    load_model = pickle.load(open(path + 'House_price_prediction_model.pkl','rb'))
    bedroom = int(input("Enter The Number of Bedrooms:"))
    bathroom = int(input("Enter The Number of Bathrooms:"))
    area = int(input("Enter The Area of House(in sqft):"))
    #fit all data into house dataFrame
    house= [[bedroom,bathroom,area]]
    pred = load_model.predict(house)
    print(f'Price of House is:{int(pred[0])}$')
    


print("******************************************************************")
print("*******************What Do you want to perform?*******************")
print("**                                                              **")
print("**1)Student Marks Prediction.                                   **")
print("**2)Iris Species Prediction.                                    **")
print("**3)Breast Cancer Prediction.                                   **")
print("**4)Sapm Mail Prediction.                                       **")
print("**5)Titanic Survival Prediction.                                **")
print("**6)Employee Salary Prediction.                                 **")
print("**7)Customer Next Purchase Prediction.                          **")
print("**8)House Price Prediction                                      **")
print("**                                                              **")
print("******************************************************************")


print('\n')
#define main body
def main():
    option = int(input("Enter Your Choice: "))
    if option == 1:
        print("**Student Marks Prediction**")
        print("This Model Predict Marks on the basis of study time\n")
        Marks()
    elif option == 2:
        print("**Iris Species Prediction**")
        print("This Model Predict Species On the Basis of 'sepal lenght,sepal width,petal length,petal width'")
        species()
    elif option == 3:
        print("**Breast Cancer Prediction**")
        print("This Model Predict Cancer on the Basis of '30-characterstics' of Cancer")
        cancer()
    elif option == 4:
        print("**Spam Mail Prediction**")
        spamham()
    elif option == 5:
        print("**Titanic Survival Prediction**")
        survival()
    elif option == 6:
        print("**Employee Salary Prediction**")
        salary()
    elif option == 7:
        print("**Customer Next Purchase Prediction**")
        purchase()
    elif option == 8:
        print("**House Price Prediction**")
        house()
    else:
        print("Enter Right option!!!!!!!!")

#call main() for run main body once
main()
#loop for repeatation
while True:
    print('\n')
    print("Do you want to Check More Models?")
    choice = input("Enter Yes for 'y' & No for 'n' :")
    choice=choice.upper()
    if choice == 'Y':
        main()
    elif choice == 'N':
        print("********************Thank You********************")
        break
    else:
        print("Please Enter Correct Choice!!")







    
