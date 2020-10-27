from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
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
genes = spark.textFile(('GeneMetaData-10-10.txt'))
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

patients = spark.textFile('PatientMetaData-10-10.txt')
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

geo = spark.textFile("GEO-10-10.txt")
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
    _      _             _ _        ___     _                  _   _             ___ _         __  __ 
   /_\  __| |_ _  _ __ _| | |_  _  |_ _|_ _| |_ ___ _ _ ___ __| |_(_)_ _  __ _  / __| |_ _  _ / _|/ _|
  / _ \/ _|  _| || / _` | | | || |  | || ' \  _/ -_) '_/ -_|_-<  _| | ' \/ _` | \__ \  _| || |  _|  _|
 /_/ \_\__|\__|\_,_\__,_|_|_|\_, | |___|_||_\__\___|_| \___/__/\__|_|_||_\__, | |___/\__|\_,_|_| |_|  
                             |__/                                        |___/                          
'''
g = sql.sql("SELECT p.patientid, p.disease, e.geneid, e.exValue " +
            "FROM patients AS p, geo AS e " +
            "WHERE p.disease =18 AND p.patientid = e.patientid")

# Gets the expression values for each gene of each patient
g1=g.groupBy('patientid').pivot('geneid').sum('exValue')

# Function to parse rows
def parseFunc(x):
    return Vectors.dense(x[1:])

parsedData = g1.rdd.map(parseFunc)
# parsedData.foreach(lambda row: print(row))

pearsonCorr = Statistics.corr(parsedData)

print(str(pearsonCorr).replace('nan', 'NaN'))

print("\nENDING PYTHON SCRIPT\n")
