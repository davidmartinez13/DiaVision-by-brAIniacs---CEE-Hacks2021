"""
CEE Hackathon
Maintainer = "ashwin.nedungadi@tu-dortmund.de
Script for predicting patient ulcer temperature
"""
import json
import ast

def input_info():
    """Input promt for user, information about patient. Returns - dict"""

    name = input("Enter Patient Full Name: ")
    age = input("Enter Patient Age: ")
    ulcer_temp = input("Enter Temperature of Ulcer: ")
    body_temp = input("Enter Temperature of Body: ")
    day = input("Enter Day: ")

    patient_info = {'Name' : name,'Age' : age, 'Ulcer Temp' : ulcer_temp, 'Body Temp' : body_temp, 'Day' : day}

    return patient_info

def create_new_patient():
    """
    Create a new patient record.
    """
    records = open("patient_records.txt", "a")
    patient = input_info()
    records.write(json.dumps(patient))
    records.write("\n")
    records.close()
    print("Record Created: ", patient)

def predict_ulcer_stage(data):
    """ Basic function to predict the ulcer's stage."""

    ulcer_temp = list()
    body_temp = list()

    for record in data:
        r = ast.literal_eval(record)
        ulcer_temp.append(r["Ulcer Temp"])
        body_temp.append(r["Body Temp"])
    #print(ulcer_temp, body_temp)

    for i in range(len(ulcer_temp)):
        u_temp = float(ulcer_temp[i])
        b_temp = float(body_temp[i])

        if u_temp > b_temp:
            temp_diff = u_temp-b_temp
            print("Temperature of wound is hotter, and by:", temp_diff)

        elif b_temp > u_temp:
            temp_diff = b_temp-u_temp
            print("Temperature of wound is colder, and by:", temp_diff )
            if temp_diff > 2:
                print("Wound critical, difference is greater than 2 degrees!")




def search():
    """ when called, returns the name to be searched in exisitng records """
    search_name = input("Enter Patient's Name: ")
    return(search_name)

def read_patient():
    """ Given a patient name, retrieves & returns all the data records of patient"""
    name = search()
    records = open("patient_records.txt", "r")

    patient_data = records.readlines()

    results = list()
    for patient in patient_data:
        p = ast.literal_eval(patient)
        if p["Name"] == name.lower():
            results.append(patient)
            print(patient)
    return results


    


def quit():
    raise SystemExit

def main():
    print("---------------------------------- PATIENT ULCER-TEMPERATURE APPLICATION ----------------------------------")
    ans = True
    while ans:
        print("""
        1.Add a Patient
        2.Modify a Patient
        3.Look Up Patient Record
        4.Predict Ulcer Stage
        5.Exit/Quit
        """)
        print("-------------------------------------------------------------------------------------------------------")
        ans = input("What would you like to do? ")
        if ans == "1":
            create_new_patient()

        elif ans == "2":
            print("\n Patient Record Updated")

        elif ans == "3":
            print("\n Report")
            patient_data = read_patient()

        elif ans == "4":
            data = read_patient()
            predict_ulcer_stage(data)

        elif ans == "5":
            print("\n Goodbye")
            quit()

        elif ans != "":
            print("\n Not Valid Choice Try again")

main()