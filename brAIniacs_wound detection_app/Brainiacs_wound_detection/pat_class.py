import json
import ast

class Patient():
    def __init__(self):
        self.Name = None
        self.Age = None
        self.Ulcer_temp = None
        self.Body_temp = None
        self.Day = None
        self.patient_info = {}

    def input_info(self,instance):
        """Input promt for user, information about patient. Returns - dict"""

        self.patient_info = {'Name': self.Name, 'Age': self.Age, 'Ulcer Temp': self.Ulcer_temp, 'Body Temp': self.Body_temp, 'Day': self.Day}

    def create_new_patient(self):
        """
        Create a new patient record.
        """
        records = open("patient_records.txt", "a")
        patient = self.patient_info
        records.write(json.dumps(patient))
        records.write("\n")
        records.close()
        print("Record Created: ", patient)

    def read_patient(self):
        """ Given a patient name, retrieves & returns all the data records of patient"""
        name = self.Name

        records = open("patient_records.txt", "r")

        patient_data = records.readlines()

        results = list()
        for patient in patient_data:
            p = ast.literal_eval(patient)
            if p["Name"] == name.lower():
                results.append(patient)
                print(patient)
        return results

    def predict_ulcer_stage(self,data):
        """ Basic function to predict the ulcer's stage."""

        ulcer_temp = list()
        body_temp = list()

        for record in data:
            r = ast.literal_eval(record)
            ulcer_temp.append(r["Ulcer Temp"])
            body_temp.append(r["Body Temp"])
        # print(ulcer_temp, body_temp)

        for i in range(len(ulcer_temp)):
            u_temp = float(ulcer_temp[i])
            b_temp = float(body_temp[i])

            if u_temp > b_temp:
                temp_diff = u_temp - b_temp
                print("Temperature of wound is hotter, and by:", temp_diff)

            elif b_temp > u_temp:
                temp_diff = b_temp - u_temp
                print("Temperature of wound is colder, and by:", temp_diff)
                if temp_diff > 2:
                    print("Wound critical, difference is greater than 2 degrees!")




