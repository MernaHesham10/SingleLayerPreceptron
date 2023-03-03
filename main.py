### Importing Packages ###
from tkinter import *
from tkinter import messagebox
import tkinter as tk
from tkinter.ttk import Combobox
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt


'''
    df.replace({"species": {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}}, inplace=True)
#bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, gender, species

'''

# Creating Object From Tkinter
tkinterWindow = Tk()

def DataPreprocessing():
    # Loading Penguins DataSet
    penguinsLoadedData = pd.read_csv('penguins.csv')

    # Handlign NaN Value That Found In Gender Column
    penguinsLoadedData["gender"].fillna("Not Identified", inplace=True)

    # label Encoding For Species Column
    #penguinsLoadedData.replace({"species": {'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2}}, inplace=True)

    # label Encoding For Gender Column
    penguinsLoadedData.replace({"gender": {'male': 2, 'female': 1, 'Not Identified': 0}}, inplace=True)

    return penguinsLoadedData


def ProcessingUserDataInput(firstFeature, secondFeature, firstClass, secondClass, addBias, numberOfEpochs, learningRate):

    def ChooseAndShuffle(firstClass, secondClass):
        firstClassTemp = firstClass
        secondClassTemp = secondClass

        penguinsLoadedData = DataPreprocessing()
        penguinsLoadedData.to_csv('EncodingData.csv', index=False)

        speciesClass = ''
        readingLines = open('EncodingData.csv').readlines()
        i = 0
        readingLinesRecords = list()

        for readingEachLine in readingLines:

            if i == 0:
                # remove heading labels
                i = i + 1
            else:
                bill_length_mm = (readingEachLine.split(','))[0]
                bill_depth_mm = (readingEachLine.split(','))[1]
                flipper_length_mm = (readingEachLine.split(','))[2]
                gender = (readingEachLine.split(','))[3]
                body_mass_g = (readingEachLine.split(','))[4]
                species = (readingEachLine.split(','))[5]

                if species == 'Adelie\n':
                    speciesClass = "Adelie"
                if species == 'Gentoo\n':
                    speciesClass = 'Gentoo'
                if species == 'Chinstrap\n':
                    speciesClass = 'Chinstrap'

                recordsDict = dict(bill_length_mm=bill_length_mm, bill_depth_mm=bill_depth_mm, flipper_length_mm=flipper_length_mm, gender=gender, body_mass_g=body_mass_g, species=speciesClass, Index=i)

                readingLinesRecords.append(recordsDict)
                readingLinesMod = i % 50
                i = i + 1
                recordsDictLine = recordsDict["bill_length_mm"] + ',' + recordsDict["bill_depth_mm"] + ',' + recordsDict["flipper_length_mm"] + ',' + recordsDict["gender"] + ',' + recordsDict[
                    "body_mass_g"] + ',' + recordsDict["species"] + '\n'  # to change the format of the dict
                if ((readingLinesMod < 31 and readingLinesMod != 0) and (speciesClass == firstClassTemp or speciesClass == secondClassTemp)):
                    with open("train.txt", "a") as trainfile:
                        trainfile.writelines(recordsDictLine)

                elif ((readingLinesMod >= 31 or readingLinesMod == 0) and (speciesClass == firstClassTemp or speciesClass == secondClassTemp)):
                    with open("test.txt", "a") as trainfile:
                        trainfile.writelines(recordsDictLine)

        with open("train.txt") as trainfile:
            lines = trainfile.readlines()
        random.shuffle(lines)
        with open("train.txt", "w") as trainfile:
            trainfile.writelines(lines)


    def ChooseFeatures(firstFeature, secondFeature, firstClass, secondClass, Filename):
        firstFeature = list()
        secondFeature = list()
        speciesClass = list()
        speciesClassNew = list()
        dataLines = open(Filename).readlines()

        for line in dataLines:

            bill_length_mm = (line.split(','))[0]
            bill_depth_mm = (line.split(','))[1]
            flipper_length_mm = (line.split(','))[2]
            gender = (line.split(','))[3]
            body_mass_g = (line.split(','))[4]
            species = (line.split(','))[5]

            if species == firstClass + '\n':
                speciesClassNew = 1
            elif species == secondClass + '\n':
                speciesClassNew = -1

            if (firstFeature == "bill_length_mm"):
                firstFeature.append(float(bill_length_mm))
            if (firstFeature == "bill_depth_mm"):
                firstFeature.append(float(bill_depth_mm))
            if (firstFeature == "flipper_length_mm"):
                firstFeature.append(float(flipper_length_mm))
            if (firstFeature == "gender"):
                firstFeature.append((gender))
            if (firstFeature == "body_mass_g"):
                firstFeature.append(float(body_mass_g))

            if (secondFeature == "bill_length_mm"):
                secondFeature.append(float(bill_length_mm))
            if (secondFeature == "bill_depth_mm"):
                secondFeature.append(float(bill_depth_mm))
            if (secondFeature == "flipper_length_mm"):
                secondFeature.append(float(flipper_length_mm))
            if (secondFeature == "gender"):
                secondFeature.append((gender))
            if (secondFeature == "body_mass_g"):
                secondFeature.append(float(body_mass_g))

            speciesClass.append(speciesClassNew)

        return (firstFeature, secondFeature, speciesClass)


    def InputMatrixConstructor(firstFeature, secondFeature, addBias):
        matrix = np.empty([60, 3], dtype=float)

        if addBias == True:
            selectedAddBias_CheckValTemp = 1

        elif addBias == False:
            selectedAddBias_CheckValTemp = 0

        for i in range(0, 60):
            matrix[i][0] = selectedAddBias_CheckValTemp
            matrix[i][1] = firstFeature[i]
            matrix[i][2] = secondFeature[i]

        return matrix


    def WeightMatrixConstructor():
        # [3, 1] => 2 features, 1 for bias
        weightMatrix = np.empty([3, 1], dtype=float)

        weight1 = np.random.rand(1, 1)
        weight2 = np.random.rand(1, 1)
        bias = np.random.rand(1, 1)

        for i in range(0, 3):
            for j in range(0, 1):
                weightMatrix[i][j] = bias
                weightMatrix[i][j] = weight1[j]
                weightMatrix[i][j] = weight2[j]

        return weightMatrix


    def Signum(netMatrix):

        if (netMatrix > 0):
            netMatrix = 1

        elif (netMatrix == 0):
            netMatrix = 1

        else:
            netMatrix = -1

        return netMatrix


    def SingleLayerPers(matrix, weightMatrix, targetList, numOfEpochs, learningRate):

        i = 0
        for j in range(0, numOfEpochs):
            for i in range(0, 60):
                recordMatrix = np.empty([1, 3])
                recordMatrix[0][0] = matrix[i][0]
                recordMatrix[0][1] = matrix[i][1]
                recordMatrix[0][2] = matrix[i][2]

                netMatrix = np.dot(recordMatrix, weightMatrix)
                normaizedValue = Signum(netMatrix)

                if (normaizedValue != targetList):
                    lossValue = targetList[i] - normaizedValue
                    termValue = np.dot(learningRate, lossValue)
                    weightMatrix[0][0] = weightMatrix[0][0] + (np.dot(termValue, recordMatrix[0][0]))
                    weightMatrix[1][0] = weightMatrix[1][0] + (np.dot(termValue, recordMatrix[0][1]))
                    weightMatrix[2][0] = weightMatrix[2][0] + (np.dot(termValue, recordMatrix[0][2]))

        return weightMatrix


    def AnotherInputMatrixConstructor(firstFeature, secondFeature, bias):
        AnotherInputMatrix = np.empty([40, 3], dtype=float)

        if bias == True:
            biasCheck = 1

        elif bias == False:
            biasCheck = 0

        for i in range(0, 40):
            AnotherInputMatrix[i][0] = biasCheck
            AnotherInputMatrix[i][1] = firstFeature[i]
            AnotherInputMatrix[i][2] = secondFeature[i]

        return AnotherInputMatrix


    def test(anotherInputMatrix, weightMatrix, targetList):
        accuracy = 0
        firstClass_True = 0
        secondClass_True = 0
        firstClass_False = 0
        secondClass_False = 0

        confMatrix = np.empty([2, 2])

        for i in range(0, 40):
            recordMatrix2 = np.empty([1, 3])

            recordMatrix2[0][0] = anotherInputMatrix[i][0]
            recordMatrix2[0][1] = anotherInputMatrix[i][1]
            recordMatrix2[0][2] = anotherInputMatrix[i][2]

            netMatrix2 = np.dot(recordMatrix2, weightMatrix)
            normaizedValue2 = Signum(netMatrix2)

            if (targetList[i] == normaizedValue2 and targetList[i] == 1):
                firstClass_True += 1
                accuracy += 1

            elif (targetList[i] == normaizedValue2 and targetList[i] == -1):
                secondClass_True += 1
                accuracy += 1

            elif (targetList[i] != normaizedValue2 and targetList[i] == 1):
                firstClass_False += 1

            elif (targetList[i] != normaizedValue2 and targetList[i] == -1):
                secondClass_False += 1

        confMatrix[0][0] = firstClass_True
        confMatrix[0][1] = secondClass_False
        confMatrix[1][0] = firstClass_False
        confMatrix[1][1] = secondClass_True

        return ((accuracy / 40) * 100), confMatrix


    def draw_line(firstFeature, secondFeature, speciesClass, updatedweight):

        FirstFeatureFirstClass = list()
        FirstFeatureSecondClass = list()
        SecondFeatureFirstClass = list()
        SecondFeatureSecondClass = list()

        for i in range(0, 40):
            if (speciesClass[i] == 1):
                FirstFeatureFirstClass.append(firstFeature[i])
                SecondFeatureFirstClass.append(secondFeature[i])

            elif (speciesClass[i] == -1):
                FirstFeatureSecondClass.append(firstFeature[i])
                SecondFeatureSecondClass.append(secondFeature[i])

        plt.figure('Figure Output Testing')
        plt.scatter(FirstFeatureFirstClass, SecondFeatureFirstClass)
        plt.scatter(FirstFeatureSecondClass, SecondFeatureSecondClass)
        min_X = min(firstFeature)
        max_X = max(firstFeature)
        min_Y = (-(updatedweight[1] * min_X) - updatedweight[0]) / updatedweight[2]
        max_Y = (-(updatedweight[1] * max_X) - updatedweight[0]) / updatedweight[2]

        plt.plot((min_X, max_X), (min_Y, max_Y))
        plt.xlabel(firstFeature)
        plt.ylabel(secondFeature)
        #plt.savefig("Feature1 VS Feature2 Figure .jpg")
        plt.show()


    def main(firstFeature, secondFeature, firstClass, secondClass, addBias, numberOfEpochs, learningRate):

        open("train.txt", 'w').close()
        open("test.txt", 'w').close()

        ChooseAndShuffle(firstClass, secondClass)
        CalFirstFeature, CalSecondFeature, CalSpeciesClass = ChooseFeatures(firstFeature, secondFeature, firstClass, secondClass, "train.txt")

        CalFirstFeature = np.array(CalFirstFeature)
        CalSecondFeature = np.array(CalSecondFeature)

        # take 2 features from 2 classes, each class has 30 train, 20 test, so take train only
        CalFirstFeature = CalFirstFeature.reshape(60, 1)
        CalSecondFeature = CalSecondFeature.reshape(60, 1)

        inputMatrix = InputMatrixConstructor(CalFirstFeature, CalSecondFeature, addBias)
        weightMatrix = WeightMatrixConstructor()
        updatedWeightMatrix = SingleLayerPers(inputMatrix, weightMatrix, CalSpeciesClass, numberOfEpochs, learningRate)

        # for test
        CalFirstFeature_Test, CalSecondFeature_Test, CalSpeciesClass_Test = ChooseFeatures(firstFeature, secondFeature, firstClass, secondClass, "test.txt")
        anotherInputMatrix = AnotherInputMatrixConstructor(CalFirstFeature_Test, CalSecondFeature_Test, addBias)
        acuuracy, confusionMatrix = test(anotherInputMatrix, updatedWeightMatrix, CalSpeciesClass_Test)
        print('Accuracy = ', int(acuuracy), '%', '\n', 'Confusion Matrix = \n', confusionMatrix, '\n')
        DisplayingProcessDataInput(int(acuuracy), confusionMatrix)

        # for drawing line
        draw_line(CalFirstFeature_Test, CalSecondFeature_Test, CalSpeciesClass_Test, updatedWeightMatrix)


    main(firstFeature, secondFeature, firstClass, secondClass, addBias, numberOfEpochs, learningRate)


# This Is Function to Clear All Content That User Entered
def ClearAllUserDataInput():
    firstFeature_Combox.set('bill_length_mm')
    secondFeature_Combox.set('bill_length_mm')

    firstClass_Combox.set('Adelie')
    secondClass_Combox.set('Adelie')

    learningRate_Entry.delete(0, END)
    learningRate_Entry.insert(END, 0)

    numberOfEpochs_Entry.delete(0, END)
    numberOfEpochs_Entry.insert(END, 0)

    addBias_Checkbox.deselect()

    processData_Label['text'] = ''


# This Is Function To Take All Data Values That User Entered
def GettingAllUserDataInput():
    savedFirstFeature = firstFeature_Tuple.index(str(firstFeature.get()))

    savedSecondFeature = SscondFeature_Tuple.index(str(secondFeature.get()))

    savedFirstClass = firstClass_Tuple.index(str(firstClass.get()))

    savedSecondClass = secondClass_Tuple.index(str(secondClass.get()))

    savedLearningRate = float(learningRate.get())

    savedNumberOfEpochs = int(numberOfEpochs.get())

    savedAddBias = addBias.get()

    # Check The Validation Of Values That User Entered

    if (firstFeature_Tuple.index(str(firstFeature.get())) == SscondFeature_Tuple.index(str(secondFeature.get()))):
        messagebox.showerror("Error", "You Mustn't Select The Same Feature in Both")

    if (firstClass_Tuple.index(str(firstClass.get())) == secondClass_Tuple.index(str(secondClass.get()))):
        messagebox.showerror("Error", "You Mustn't Select The Same Class in Both")

    if (savedAddBias == 1):
        savedAddBiasEntered_Check = True

    if (savedAddBias == 0):
        savedAddBiasEntered_Check = False


    print (firstFeature_Tuple[savedFirstFeature], SscondFeature_Tuple[savedSecondFeature], firstClass_Tuple[savedFirstClass], secondClass_Tuple[savedSecondClass], savedAddBiasEntered_Check, savedNumberOfEpochs, savedLearningRate)

    ProcessingUserDataInput(firstFeature_Tuple[savedFirstFeature], SscondFeature_Tuple[savedSecondFeature], firstClass_Tuple[savedFirstClass], secondClass_Tuple[savedSecondClass], savedAddBiasEntered_Check, savedNumberOfEpochs, savedLearningRate)


def DisplayingProcessDataInput(acuuracy, confusionMatrix):
    processData_Label["text"] = f"Output \n Accuracy =  {int(acuuracy)} % \n Confusion Matrix =  {confusionMatrix}"


# This Is Function To Make Whole DataSet Plots
def MakingWholeDataSetPlots():

    penguinsLoadedData = DataPreprocessing()

    # Dividing DataSet Into 3 Classes According to their Species
    data_Grouped_1 = penguinsLoadedData.groupby(penguinsLoadedData.species)
    data_AdelieGroup = data_Grouped_1.get_group('Adelie')

    data_Grouped_2 = penguinsLoadedData.groupby(penguinsLoadedData.species)
    data_GentooGroup = data_Grouped_2.get_group('Gentoo')

    data_Grouped_3 = penguinsLoadedData.groupby(penguinsLoadedData.species)
    data_ChinstrapGroup = data_Grouped_3.get_group('Chinstrap')

    plt.figure('Figure bill_length_mm VS bill_depth_mm Features')
    plt.title("bill_length_mm VS bill_depth_mm Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['bill_depth_mm'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['bill_depth_mm'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['bill_depth_mm'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('bill_depth_mm')
    # plt.savefig("bill_length_mm vs bill_depth_mm Figure .jpg")
    plt.show()

    plt.figure('Figure bill_length_mm VS flipper_length_mm Features')
    plt.title("bill_length_mm VS flipper_length_mm Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['flipper_length_mm'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['flipper_length_mm'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['flipper_length_mm'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('flipper_length_mm')
    # plt.savefig("bill_length_mm vs flipper_length_mm .jpg")
    plt.show()

    plt.figure('Figure bill_length_mm VS body_mass_g Features')
    plt.title("bill_length_mm VS body_mass_g Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['body_mass_g'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['body_mass_g'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['body_mass_g'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('body_mass_g')
    # plt.savefig("bill_length_mm vs body_mass_g Figure .jpg")
    plt.show()

    plt.figure('Figure bill_length_mm VS gender Features')
    plt.title("bill_length_mm VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_length_mm'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['bill_length_mm'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['bill_length_mm'], data_GentooGroup['gender'])
    plt.xlabel('bill_length_mm')
    plt.ylabel('gender')
    # plt.savefig("bill_length_mm vs gender Figure .jpg")
    plt.show()

    plt.figure('Figure bill_depth_mm VS flipper_length_mm Features')
    plt.title("bill_depth_mm VS flipper_length_mm Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_depth_mm'], data_AdelieGroup['flipper_length_mm'])
    plt.scatter(data_ChinstrapGroup['bill_depth_mm'], data_ChinstrapGroup['flipper_length_mm'])
    plt.scatter(data_GentooGroup['bill_depth_mm'], data_GentooGroup['flipper_length_mm'])
    plt.xlabel('bill_depth_mm')
    plt.ylabel('flipper_length_mm')
    # plt.savefig("bill_depth_mm vs flipper_length_mm Figure .jpg")
    plt.show()

    plt.figure('Figure bill_depth_mm VS body_mass_g Features')
    plt.title("bill_depth_mm VS body_mass_g Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_depth_mm'], data_AdelieGroup['body_mass_g'])
    plt.scatter(data_ChinstrapGroup['bill_depth_mm'], data_ChinstrapGroup['body_mass_g'])
    plt.scatter(data_GentooGroup['bill_depth_mm'], data_GentooGroup['body_mass_g'])
    plt.xlabel('bill_depth_mm')
    plt.ylabel('body_mass_g')
    # plt.savefig("bill_depth_mm vs body_mass_g Figure .jpg")

    plt.show()

    plt.figure('Figure bill_depth_mm VS gender Features')
    plt.title("bill_depth_mm VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['bill_depth_mm'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['bill_depth_mm'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['bill_depth_mm'], data_GentooGroup['gender'])
    plt.xlabel('bill_depth_mm')
    plt.ylabel('gender')
    # plt.savefig("bill_depth_mm vs gender Figure .jpg")
    plt.show()

    plt.figure('Figure flipper_length_mm VS body_mass_g Features')
    plt.title("flipper_length_mm VS body_mass_g Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['flipper_length_mm'], data_AdelieGroup['body_mass_g'])
    plt.scatter(data_ChinstrapGroup['flipper_length_mm'], data_ChinstrapGroup['body_mass_g'])
    plt.scatter(data_GentooGroup['flipper_length_mm'], data_GentooGroup['body_mass_g'])
    plt.xlabel('flipper_length_mm')
    plt.ylabel('body_mass_g')
    # plt.savefig("flipper_length_mm vs body_mass_g Figure .jpg")
    plt.show()

    plt.figure('Figure flipper_length_mm VS gender Features')
    plt.title("flipper_length_mm VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['flipper_length_mm'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['flipper_length_mm'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['flipper_length_mm'], data_GentooGroup['gender'])
    plt.xlabel('flipper_length_mm')
    plt.ylabel('gender')
    # plt.savefig("flipper_length_mm vs gender Figure .jpg")
    plt.show()

    plt.figure('Figure body_mass_g VS gender Features')
    plt.title("body_mass_g VS gender Features", size=15, color="red", pad=15)
    plt.scatter(data_AdelieGroup['body_mass_g'], data_AdelieGroup['gender'])
    plt.scatter(data_ChinstrapGroup['body_mass_g'], data_ChinstrapGroup['gender'])
    plt.scatter(data_GentooGroup['body_mass_g'], data_GentooGroup['gender'])
    plt.xlabel('body_mass_g')
    plt.ylabel('gender')
    # plt.savefig("body_mass_g vs gender Figure .jpg")
    plt.show()



mainTitleApp_Label = tk.Label(tkinterWindow, text ="Single Layer Perceptron Application", font=('Lucida Calligraphy', 13), padx=20, pady=10, fg='blue')
mainTitleApp_Label.grid(column=1, row=1)

##################### Select The First Feature ########################
firstFeature_Label = tk.Label(tkinterWindow, text="Select The First Feature", font=('Lucida Calligraphy', 10), pady=10)
firstFeature_Label.grid(column=1, row=16)

firstFeature_Tuple = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')

firstFeature = StringVar()
firstFeature_Combox = Combobox(tkinterWindow, textvariable = firstFeature, font=('Lucida Calligraphy', 10))
firstFeature_Combox['values'] = firstFeature_Tuple
firstFeature_Combox['state'] = 'readonly'
firstFeature_Combox.set('bill_length_mm')
firstFeature_Combox.grid(column = 2, row = 16)

##################### Select The Second Feature ########################
secondFeature_Label = tk.Label(tkinterWindow, text="Select The Second Feature", font=('Lucida Calligraphy', 10), pady=10)
secondFeature_Label.grid(column = 1, row = 20)

SscondFeature_Tuple = ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'gender', 'body_mass_g')

secondFeature = StringVar()
secondFeature_Combox = Combobox(tkinterWindow, textvariable=secondFeature, font=('Lucida Calligraphy', 10))
secondFeature_Combox['values'] = SscondFeature_Tuple
secondFeature_Combox['state'] = 'readonly'
secondFeature_Combox.set('bill_length_mm')
secondFeature_Combox.grid(column= 2, row=20)

##################### Select The First Class ########################
firstClass_Label = tk.Label(tkinterWindow, text="Select The First Class", font=('Lucida Calligraphy', 10), pady=10)
firstClass_Label.grid(column=1, row=30)

firstClass_Tuple = ('Adelie', 'Gentoo', 'Chinstrap')

firstClass = StringVar()
firstClass_Combox = Combobox(tkinterWindow, textvariable=firstClass, font=('Lucida Calligraphy', 10))
firstClass_Combox['values'] = firstClass_Tuple
firstClass_Combox['state'] = 'readonly'
firstClass_Combox.set('Adelie')
firstClass_Combox.grid(column= 2, row=30)

##################### Select The Second Class ########################
secondClass_Label = tk.Label(tkinterWindow, text="Select The Second Class", font=('Lucida Calligraphy', 10), pady=10)
secondClass_Label.grid(column=1, row=36)

secondClass_Tuple = ('Adelie', 'Gentoo', 'Chinstrap')

secondClass = StringVar()
secondClass_Combox = Combobox(tkinterWindow, textvariable=secondClass, font=('Lucida Calligraphy', 10))
secondClass_Combox['values'] = secondClass_Tuple
secondClass_Combox['state'] = 'readonly'
secondClass_Combox.set('Adelie')
secondClass_Combox.grid(column= 2, row=36)

####################### Enter Learning Rate (eta) ############################
learningRate = tk.StringVar()
learningRate_Label = tk.Label(tkinterWindow, text="Enter Learning Rate (ETA)", font=('Lucida Calligraphy', 10), pady=10)
learningRate_Entry = Entry(tkinterWindow, textvariable=learningRate, font=('Lucida Calligraphy', 10))
learningRate_Entry.insert(END, 0)

learningRate_Label.grid(column=1, row=60)
learningRate_Entry.grid(column=2, row=60)

####################### Enter Number Of Epochs (m) ############################
numberOfEpochs = tk.StringVar()
numberOfEpochs_Label = tk.Label(tkinterWindow, text="Enter Number Of Epochs (m)", font=('Lucida Calligraphy', 10), pady=10)
numberOfEpochs_Entry = Entry(tkinterWindow, textvariable=numberOfEpochs, font=('Lucida Calligraphy', 10))
numberOfEpochs_Entry.insert(END, 0)

numberOfEpochs_Label.grid(column=1, row=53)
numberOfEpochs_Entry.grid(column=2, row=53)

################# ADD Bias ########################
addBias = IntVar()
addBias_Checkbox = Checkbutton(tkinterWindow, text="Add Bias", variable=addBias, font=('Lucida Calligraphy', 10), pady=10, padx=50)
addBias_Checkbox.grid(column=2, row=66)

######################## Show Output Result ################################
processData_Label = tk.Label(master=tkinterWindow, font=('Lucida Calligraphy', 10), pady=10)
processData_Label.place(x=200, y=350)

###################### Buttons ###############################
makePlots_Btn = Button(tkinterWindow, text='Show Data Plots', command=MakingWholeDataSetPlots, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
makePlots_Btn.place(x=210, y=450)

processData_Btn = Button(tkinterWindow, text='Process Inputs', command= GettingAllUserDataInput, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
processData_Btn.place(x=220, y=520)

ClearInputs_Btn = Button(tkinterWindow, text="Clear Inputs", command= ClearAllUserDataInput, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
ClearInputs_Btn.place(x=225, y=590)

QuitApp_Btn = Button(tkinterWindow, text='Quit Application', command=tkinterWindow.quit, font=('Lucida Calligraphy', 11), pady=10, padx=50, bg='white', fg='blue')
QuitApp_Btn.place(x=205, y=660)

########################### Main #############################
tkinterWindow.title('Single Layer Perceptron Application')
tkinterWindow.geometry("650x730")
tkinterWindow.mainloop()