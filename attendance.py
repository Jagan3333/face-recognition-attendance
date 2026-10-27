
import csv
import os
from datetime import datetime

ATTENDANCE_FILE = "attendance.csv"


def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    file_exists = os.path.exists(ATTENDANCE_FILE)

    with open(ATTENDANCE_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])

        writer.writerow([name, today, current_time])

    print(f"Attendance saved for {name} at {current_time}")


if __name__ == "__main__":
    person_name = input("Enter the recognized person's name: ")

    if person_name.strip():
        mark_attendance(person_name.strip())
    else:
        print("Name cannot be empty.")
