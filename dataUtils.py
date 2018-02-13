"""

>50K, <=50K.

age: continuous.
workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
fnlwgt: continuous.
education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
education-num: continuous.
marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
sex: Female, Male.
capital-gain: continuous.
capital-loss: continuous.
hours-per-week: continuous.
native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

"""

import random
import tensorflow as tf

WORKCLASS = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "?"]
MAXFNLWGT = 1000000
EDUCATION = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool", "?"]
MAXEDUCATIONNUM = 16
MARITALSTATUS = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse", "?"]
OCCUPATION = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces", "?"]
RELATIONSHIP = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried", "?"]
RACE = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black", "?"]
SEX = ["Female", "Male", "?"]
MAXCAPITALGAIN = 99999
MAXCAPITALLOSS = 99999
MAXHOURSPERWEEK = 99
NATIVECOUNTRY = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands", "?"]
INCOME = [">50K", "<=50K"]


"""
returns two lists of data
the data is size [record_count , 113]
the labels is size [record_count, 2]

recommended usage
trainingdata, traininglabels = readData("project1datatraining.csv")
testdata, testlabels = readData("project1datatest.csv")
"""
def readData(filename):
    with open(filename, "r") as f:
        data_string = f.read()
    data_string = data_string.splitlines()
    data = map(formatRow, map(lambda x: x.split(","), data_string))

    data, labels = zip(*data)

    return data, labels

def oneHot(index, count):
    return [1.0 if x == index else 0.0 for x in range(count)]

def formatRow(row):

    newrow = []

    # age
    age = int(row[0])
    newrow.append((age - 50.0)/50.0)

    # workclass
    workclass = row[1]
    index = WORKCLASS.index(workclass.strip())
    newrow.extend(oneHot(index, len(WORKCLASS)))

    # fnlwgt final samplling weight
    fnlwgt = int(row[2])
    newrow.append((fnlwgt - (MAXFNLWGT/2.0))/(MAXFNLWGT/2.0))

    # education
    education = row[3]
    index = EDUCATION.index(education.strip())
    newrow.extend(oneHot(index, len(EDUCATION)))

    # education-num
    education_num = int(row[4])
    newrow.append((education_num - (MAXEDUCATIONNUM/2.0))/(MAXEDUCATIONNUM/2.0))

    # marital-status
    marital_status = row[5]
    index = MARITALSTATUS.index(marital_status.strip())
    newrow.extend(oneHot(index, len(MARITALSTATUS)))

    # occupation
    occupation = row[6]
    index = OCCUPATION.index(occupation.strip())
    newrow.extend(oneHot(index, len(OCCUPATION)))

    # relationship
    relationship = row[7]
    index = RELATIONSHIP.index(relationship.strip())
    newrow.extend(oneHot(index, len(RELATIONSHIP)))

    # race
    race = row[8]
    index = RACE.index(race.strip())
    newrow.extend(oneHot(index, len(RACE)))

    # sex
    sex = row[9]
    index = SEX.index(sex.strip())
    newrow.extend(oneHot(index, len(SEX)))

    # capital-gains
    capital_gains = int(row[10])
    newrow.append((capital_gains - (MAXCAPITALGAIN / 2.0)) / (MAXCAPITALGAIN / 2.0))

    # capital-loss
    capital_loss = int(row[11])
    newrow.append((capital_loss - (MAXCAPITALLOSS / 2.0)) / (MAXCAPITALLOSS / 2.0))

    # hours-per-week
    hours_per_week = int(row[12])
    newrow.append((hours_per_week - (MAXHOURSPERWEEK / 2.0)) / (MAXHOURSPERWEEK / 2.0))

    # native-country
    native_country = row[13]
    index = NATIVECOUNTRY.index(native_country.strip())
    newrow.extend(oneHot(index, len(NATIVECOUNTRY)))

    # income label
    income = row[14]
    income = oneHot(INCOME.index(income.strip()), len(INCOME))

    return newrow, income

"""
Takes the full dataset and take a random sample of 100
Make sure that the 100 samples of data still match with the labels when done
"""
def getBatch(data, labels, batch_size):
    return zip(*random.sample(list(zip(data, labels)), batch_size))


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
