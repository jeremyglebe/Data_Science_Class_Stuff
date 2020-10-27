import time
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionWithSGD
from pyspark.mllib.regression import LinearRegressionModel
import pyspark.mllib.regression as regr

'''
  ___      _             
 / __| ___| |_ _  _ _ __ 
 \__ \/ -_)  _| || | '_ \
 |___/\___|\__|\_,_| .__/
                   |_|   
'''
# Create a spark context from which to do sparky things
spark = SparkContext(appName="App")
print("\nSTARTING PYTHON SCRIPT\n")
# Create an SQL context to perform SQL operations on files
sql = SQLContext(spark)

'''
  _____     _      ___   _____     _    _     
 |_   _|_ _| |_ __|_  )_|_   _|_ _| |__| |___ 
   | | \ \ /  _|___/ /___|| |/ _` | '_ \ / -_)
   |_| /_\_\\__|  /___|   |_|\__,_|_.__/_\___|
'''
# Open genes metadata with spark
genes = spark.textFile(('GeneMetaData-100-100.txt'))
header = genes.first()  # Extract header
# Extract rows from genes that are not the header
genes = genes.filter(lambda x: x != header)
# Split each row of genes by commas
gparts = genes.map(lambda l: l.split(", "))
# Now each gene item is described as a list
# gparts.foreach(lambda x: print(x))
# Rows where each row comes from select data in gparts' lists
# 'p' in lambda represents an individual list
geneframe = gparts.map(lambda p: Row(
    geneid=int(p[0]),  # Each column of our row corresponds w/ an indexed value
    target=int(p[1]),
    # Book said "long", but "long" doesn't exist, use "float" to parse then "int"
    position=int(float(p[2])),
    length=int(p[3]),
    function=int(p[4])
))
# Display the row objects
# geneframe.foreach(lambda x: print(x))
# Create an sql data frame from the rows
schemaGene = sql.createDataFrame(geneframe)
schemaGene.registerTempTable("genes")  # is sql this will be table "genes"
# print("Genes")
# schemaGene.foreach(lambda row: print(row))

patients = spark.textFile('PatientMetaData-100-100.txt')
header = patients.first()
patients = patients.filter(lambda x: x != header)
pparts = patients.map(lambda l: l.split(", "))
patientsframe = pparts.map(lambda p: Row(
    patientid=int(p[0]),
    age=int(p[1]),
    gender=int(p[2]),
    zipcode=int(p[3]),
    disease=int(p[4]),
    drugResponse=float(p[5])
))
schemaPatients = sql.createDataFrame(patientsframe)
schemaPatients.registerTempTable("patients")
# print("\nPatients")
# schemaPatients.foreach(lambda row: print(row))

geo = spark.textFile("GEO-100-100.txt")
header = geo.first()
geo = geo.filter(lambda x: x != header)
geoparts = geo.map(lambda l: l.split(", "))
geoframe = geoparts.map(lambda p: Row(
    geneid=int(p[0]),
    patientid=int(p[1]),
    exValue=float(p[2])
))
schemaGEO = sql.createDataFrame(geoframe)
schemaGEO.registerTempTable("geo")
# print("\nMicroarray Data")
# schemaGEO.foreach(lambda row: print(row))

'''
  ___  ___  _       ___          
 / __|/ _ \| |     / _ \ _ __ ___
 \__ \ (_) | |__  | (_) | '_ (_-<
 |___/\__\_\____|  \___/| .__/__/
                        |_|      
'''
g = sql.sql("SELECT p.patientid, p.disease, e.geneid, e.exValue, p.drugResponse " +  # data we're extracting
            "FROM genes AS g, patients AS p, geo AS e " +  # aliases for rows
            "WHERE g.function < 300 AND g.geneid = e.geneid " +  # gene specifiers
            "AND p.patientid = e.patientid")  # associate expression values with their patient ids only
g.registerTempTable("responses")
# print("\nResponses")
# g.foreach(lambda row: print(row))

# Get the expression values for each type of gene for each patient
# (sum is getting the sum of ONE value)
g2 = g.groupBy('patientid').pivot('geneid').sum('exValue')
g2.registerTempTable('gen')
# print("\nGen")
# g2.foreach(lambda row: print(row))

# Get a selection of patient meta data
g3 = sql.sql("SELECT patientid, disease, drugResponse FROM patients")
g3.registerTempTable("gen3")
# print("\nGen3")
# g3.foreach(lambda row: print(row))

# For each patient, combine their g3 data (disease, drug response) with
# their genes' expression values (g2) by their patientid, found in both
g4 = sql.sql("SELECT * FROM gen3, gen WHERE gen3.patientid=gen.patientid")
# print("\nNicely formatted patient data")
# g4.foreach(lambda row: print(row))

# Parse out the patient id now b/c we don't really need it.
# x[2] is the drug response, x[4:] is all the gene/expression pairs
# AttributeError: 'DataFrame' object has no attribute 'map', fixed by adding .rdd
parsedData = g4.rdd.map(lambda x: LabeledPoint(x[2], x[4:]))
# print("\nFinal Data")
# parsedData.foreach(lambda row: print(row))

'''
  ___                        _            __  __         _     _ 
 | _ \___ __ _ _ _ ___ _____(_)___ _ _   |  \/  |___  __| |___| |
 |   / -_) _` | '_/ -_|_-<_-< / _ \ ' \  | |\/| / _ \/ _` / -_) |
 |_|_\___\__, |_| \___/__/__/_\___/_||_| |_|  |_\___/\__,_\___|_|
         |___/                                                   
'''
# Build the model
# drug response is the value we are predicting, expression values are the inputs
print("Training the regression model...")
start_time = time.time()
# model = LinearRegressionWithSGD.train(parsedData)
model = LinearRegressionWithSGD.train(parsedData, step=0.000000000000015, iterations=1000)
print(f'Finished in {time.time() - start_time}s...')

# Evaluate the model on training data
# label is drug reponse, features is expression values
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))

print("Values & Predictions")
valuesAndPreds.foreach(lambda row: print(row))

print("Calculating Mean Squared Error...")
start_time = time.time()
MSE = valuesAndPreds.map(lambda t: (
    t[0]-t[1])**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
print(f'Finished in {time.time() - start_time}s...')
print(f"Mean Squared Error = {str(MSE)}")

print("\nENDING PYTHON SCRIPT\n")
