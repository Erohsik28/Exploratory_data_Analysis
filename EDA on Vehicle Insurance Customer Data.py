
# EDA on Vehicle Insurance Customer Data

# importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## 1.  Adding the column names to both datasets while reading them:

## The "customer details" file is read and is given the column headings

table1 = pd.read_csv("customer_details.csv")
table1.columns=["Customer_id","Gender","Age","Driving_license_present","Region_Code","Previously_insured","Vehicle_age","Vehicle_damage"]

#To look at the customer details table
table1


## The "customer_policy_details.csv file is read and the column names are given


table2 = pd.read_csv("customer_policy_details.csv")
table2.columns=["Customer_id", "Annual_premium", "Sales_Channel_Code", "Vintage", "Response" ]


#To look at the customer policy details table
table2


## 2. Checking and Cleaning Data Quality:

## i.Scanning for Null values and cleaning if any present


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For table1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Generating a summary of count of all the null values column wise for table 1

table1.isnull().sum()


# Drop Null values for customer_id because central tendencies for id’s is not feasible

table1 = table1[~table1.Customer_id.isnull()].copy()


table1.isnull().sum()


table1



# Replace all null values for numeric columns by mean for table 1

#numeric data
age_mean = table1['Age'].mean()
table1['Age'].fillna(value=age_mean,inplace=True)

region_mean = table1['Region_Code'].mean()
table1['Region_Code'].fillna(value=region_mean,inplace=True)


# Replace all null values for Categorical value by mode for table 1

#categorical data

gender_mode=table1.Gender.mode()[0]
table1.Gender.fillna(gender_mode,inplace=True)

driving_license_present_mode=table1.Driving_license_present.mode()[0]
table1.Driving_license_present.fillna(driving_license_present_mode,inplace=True)

previously_insured_mode=table1.Previously_insured.mode()[0]
table1.Previously_insured.fillna(previously_insured_mode,inplace=True)

vehicle_damage_mode=table1.Vehicle_damage.mode()[0]
table1.Vehicle_damage.fillna(vehicle_damage_mode,inplace=True)

vehicle_age_mode=table1.Vehicle_age.mode()[0]
table1.Vehicle_age.fillna(vehicle_age_mode,inplace=True)


table1.isnull().sum()



table1.describe()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # For table2:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Generating a summary of count of all the null values column wise

table2.isnull().sum()


# Drop Null values for customer_id because central tendencies for id’s is not feasible

table2 = table2[~table2.Customer_id.isnull()].copy()

table2.isnull().sum()



# Replace all null values for numeric columns by mean for table 2

#numeric data
annual_mean = table2['Annual_premium'].mean()
table2['Annual_premium'].fillna(value=annual_mean,inplace=True)

sales_channel_mean = table2['Sales_Channel_Code'].mean()
table2['Sales_Channel_Code'].fillna(value=sales_channel_mean,inplace=True)

vintage_mean = table2['Vintage'].mean()
table2['Vintage'].fillna(value=vintage_mean,inplace=True)


# Replace all null values for Categorical value by mode for table 2

#categorical data

response_mode=table2.Response.mode()[0]
table2.Response.fillna(response_mode,inplace=True)


table2.isnull().sum()


table2.describe()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Outliers detection For table1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Age Outliers detection

x= table1.describe()["Age"]

IQR_age= x["75%"]-x["25%"]

table1[table1["Age"]<x["25%"]-1.5*IQR_age]

table1[table1["Age"]>x["75%"]+1.5*IQR_age]



# Region Code Outlier detection

y= table1.describe()["Region_Code"]

IQR_rc= y["75%"]-y["25%"]

table1[table1["Region_Code"]<x["25%"]-1.5*IQR_rc]

table1[table1["Region_Code"]>x["75%"]+1.5*IQR_rc]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Outliers detection For table2:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Annual Premium outlier detection 

x= table2.describe()["Annual_premium"]

IQR_ap= x["75%"]-x["25%"]

table2[table2["Annual_premium"]<x["25%"]-1.5*IQR_ap]

table2[table2["Annual_premium"]>x["75%"]+1.5*IQR_ap]

mean = table2["Annual_premium"].mean()
upper_outliers = table2["Annual_premium"]>x["75%"]+1.5*IQR_ap

mean

table2.loc[upper_outliers,"Annual_premium"]=np.nan
table2["Annual_premium"].fillna(mean,inplace=True)

table2[table2["Annual_premium"]>x["75%"]+1.5*IQR_ap]


# Sales Channel Code Outlier detection

x= table2.describe()["Sales_Channel_Code"]

IQR_scc= x["75%"]-x["25%"]

table2[table2["Sales_Channel_Code"]<x["25%"]-1.5*IQR_scc]

table2[table2["Sales_Channel_Code"]>x["75%"]+1.5*IQR_scc]


# Vintage Outlier Detection

x= table2.describe()["Vintage"]

IQR_v= x["75%"]-x["25%"]

table2[table2["Vintage"]<x["25%"]-1.5*IQR_v]


table2[table2["Vintage"]>x["75%"]+1.5*IQR_v]


# iii. Removing White spaces

table1.head()


table1['Vehicle_age'].str.replace(' ','')
table1['Gender'].str.replace(' ','')
table1['Vehicle_damage'].replace(' ','')
table1


# iv. case correction. Convert string values to uppercase

table1["Gender"]=table1["Gender"].str.upper()
table1["Vehicle_damage"]=table1["Vehicle_damage"].str.upper()

table1.sample(10)


# v. Convert nominal data (categorical) into dummies 

# for future modeling use if required 
 

table1_d=pd.get_dummies(table1, columns=['Gender','Vehicle_damage','Driving_license_present','Previously_insured','Vehicle_age'])


table2_d=pd.get_dummies(table2,columns=['Response'])


# vi. Droping Duplicates (duplicated rows)

table1.drop_duplicates(inplace=True)
table1_d.drop_duplicates(inplace=True) #table1 with the dummy columns


table2.drop_duplicates(inplace=True)
table2_d.drop_duplicates(inplace=True) #table2 with the dummy columns


# 3. Create a Master table for future use. Join the customer table and customer_policy table 
# to get a master table using customer_id in both tables.


table_merge=pd.merge(table1,table2,on='Customer_id')


table_merge_d=pd.merge(table1_d,table2_d,on="Customer_id") #this is a merged table for the dummies table


table_merge


table_merge_d.describe()


table_merge.describe()


# 4. Gaining insights from the data for future growth of the insurance company

# The following information are collected:

# i. Gender wise average annual premium: To find the gender who have taken the most policies from the company so we can 
# find ways to further increase the number

table_merge.groupby('Gender')['Annual_premium'].mean()


# # ii. Age wise average annual premium: The result is plotted as a graph to get visualized result of the age distribution
# of the policy holder's average annual premium 

age_plot= table_merge.groupby('Age')['Annual_premium'].mean()
age_plot.plot()
plt.show()


age_plot  #age wise average annual premium


#iii. Finding if the data is balanced between the two genders

print("Proportion of Males in data: ", table_merge_d[table_merge_d["Gender_MALE"]==1].shape[0]/table_merge_d.shape[0]*100)


print("Proportion Males in data:",table_merge_d[table_merge_d["Gender_FEMALE"]==1].shape[0]/table_merge_d.shape[0]*100)


# # iv. Vehicle age wise average annual premium.

table_merge.groupby('Vehicle_age')['Annual_premium'].mean()


vehicle_age_plot= table_merge.groupby('Vehicle_age')['Annual_premium'].mean()
vehicle_age_plot.plot()
plt.show()


# # Is there any relation between Person Age and annual premium?

n = table_merge["Age"].corr(table_merge["Annual_premium"])


if n<-0.5:
    print("Strong negative relationship")
elif n>0.5:
    print("Strong positive relationship")
elif n>-0.5 and n <0.5:
    print("There is no relationship")


table_merge.to_csv('table_merge.csv',index=False)
