import os
import sys
import csv
import json
from datetime import datetime


class Saver:
    def __init__(self, args): # full path example: /content/drive/MyDrive/itt_most/egyetem/onlab/run_results/forecasting_M_ETTh1/2026_03_21_16_51_21 and inside here: /forecast_examples and /learning_curves
        # Base folder
        self.drive_path = "/content/drive/MyDrive/itt_most/egyetem/onlab/run_results"

        # check if drive is mounted
        if not os.path.isdir(self.drive_path):
            raise FileNotFoundError(f"The given drive_path does not exist: {self.drive_path}")

        self.args = args


        # Folder names
        self.current_task_folder = f"{args.task_name}_{args.features}_{args.data_name}"
        self.current_time_folder = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_name = f"{self.current_task_folder}/{self.current_time_folder}"

        # Full paths
        self.task_path = os.path.join(self.drive_path, self.current_task_folder)
        self.path_name = os.path.join(self.task_path, self.current_time_folder)

        # Subfolders
        self.forecast_examples_path = os.path.join(self.path_name, "forecast_examples")
        self.learning_curves_path = os.path.join(self.path_name, "learning_curves")

        # main path
        self.registry_path = os.path.join(self.drive_path, "run_registry.csv")


        # Create folders
        os.makedirs(self.task_path, exist_ok=True)
        try:
            os.makedirs(self.path_name, exist_ok=False)
        except FileExistsError:
            print(f"Folder already in use: {self.path_name}")
            sys.exit(1)
        os.makedirs(self.forecast_examples_path, exist_ok=True)
        os.makedirs(self.learning_curves_path, exist_ok=True)
        
        # save args locally
        self._save_args_file()

        # create initial row in global csv
        self._create_registry_entry()


        print("---------------")
        print("EXPERIMENT PATH: ", self.path_name)
        print("---------------")

    def _save_args_file(self):
        args_txt_path = os.path.join(self.path_name, "args.txt")
        with open(args_txt_path, "w", encoding="utf-8") as f:
            for key, value in sorted(vars(self.args).items()):
                f.write(f"{key}: {value}\n")

    def _create_registry_entry(self):
        file_exists = os.path.isfile(self.registry_path)
        fieldnames = self._registry_fieldnames()

        row = {
            "experiment_name": self.experiment_name,
            "message": "",
            "status": "running",
            "results": "",
            "args": json.dumps(vars(self.args), ensure_ascii=False),
            "run_path": self.path_name,
        }

        with open(self.registry_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def _save_results_files(self, results, message=""):
        results_txt_path = os.path.join(self.path_name, "results.txt")
        with open(results_txt_path, "w", encoding="utf-8") as f:
            for key, value in sorted(results.items()):
                f.write(f"{key}: {value}\n")

        results_json_path = os.path.join(self.path_name, "results.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        message_path = os.path.join(self.path_name, "message.txt")
        with open(message_path, "w", encoding="utf-8") as f:
            f.write(message)

    def _update_registry_entry(self, results, message="", status="finished"):
        fieldnames = self._registry_fieldnames()

        if not os.path.isfile(self.registry_path):
            raise FileNotFoundError(f"Registry csv not found: {self.registry_path}")

        rows = []
        found = False

        with open(self.registry_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["experiment_name"] == self.experiment_name:
                    row["message"] = message
                    row["status"] = status
                    row["results"] = json.dumps(results, ensure_ascii=False)
                    row["args"] = json.dumps(vars(self.args), ensure_ascii=False)
                    row["run_path"] = self.path_name
                    found = True
                rows.append(row)

        if not found:
            raise RuntimeError(
                f"Couldn't find experiment to update in registry: {self.experiment_name}"
            )

        with open(self.registry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_results(self, results, message=""):
        self._save_results_files(results, message=message)
        self._update_registry_entry(results=results, message=message, status="finished")

    def save_failed_run(self, message="failed"):
        failed_results = {}
        self._save_results_files(failed_results, message=message)
        self._update_registry_entry(results=failed_results, message=message, status="failed")

    def get_path(self, folder_name=None, file_name=None):
        if folder_name is None:
            base_path = self.path_name
        elif folder_name == "forecast_examples":
            base_path = self.forecast_examples_path
        elif folder_name == "learning_curves":
            base_path = self.learning_curves_path
        else:
            raise ValueError(f"Unknown folder: {folder_name}")

        if file_name is None:
            return base_path

        return os.path.join(base_path, file_name)






"""
import os
import csv

from datetime import datetime


class Saver:
    def __init__(self, args):
        self.drive_path = "/content/drive/MyDrive/itt_most/egyetem/onlab/run_results"

        if not os.path.isdir(self.drive_path):
            raise FileNotFoundError(f"A megadott drive_path nem létezik: {self.drive_path}")

        self.args = args

        # folder names
        self.current_task_folder = f"{args.task_name}_{args.features}_{args.data_name}"
        self.current_time_folder = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_name = f"{self.current_task_folder}/{self.current_time_folder}"

        # paths
        self.task_path = os.path.join(self.drive_path, self.current_task_folder)
        self.path_name = os.path.join(self.task_path, self.current_time_folder)

        self.forecast_examples_path = os.path.join(self.path_name, "forecast_examples")
        self.learning_curves_path = os.path.join(self.path_name, "learning_curves")

        self.registry_path = os.path.join(self.drive_path, "run_registry.csv")

        # create folders
        os.makedirs(self.task_path, exist_ok=True)

        try:
            os.makedirs(self.path_name, exist_ok=False)
        except FileExistsError:
            raise RuntimeError(f"Ez a mappa már használatban van: {self.path_name}")

        os.makedirs(self.forecast_examples_path, exist_ok=True)
        os.makedirs(self.learning_curves_path, exist_ok=True)

        # save args locally
        self._save_args_files()

        # create initial row in global csv
        self._create_registry_entry()

    def _registry_fieldnames(self):
        return [
            "experiment_name",
            "message",
            "status",
            "results",
            "args",
            "run_path",
        ]

    def _save_args_files(self):
        args_txt_path = os.path.join(self.path_name, "args.txt")
        with open(args_txt_path, "w", encoding="utf-8") as f:
            for key, value in sorted(vars(self.args).items()):
                f.write(f"{key}: {value}\n")

        args_json_path = os.path.join(self.path_name, "args.json")
        with open(args_json_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False)

    def _save_results_files(self, results, message=""):
        results_txt_path = os.path.join(self.path_name, "results.txt")
        with open(results_txt_path, "w", encoding="utf-8") as f:
            for key, value in sorted(results.items()):
                f.write(f"{key}: {value}\n")

        results_json_path = os.path.join(self.path_name, "results.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        message_path = os.path.join(self.path_name, "message.txt")
        with open(message_path, "w", encoding="utf-8") as f:
            f.write(message)

    def _update_registry_entry(self, results, message="", status="finished"):
        fieldnames = self._registry_fieldnames()

        if not os.path.isfile(self.registry_path):
            raise FileNotFoundError(f"A registry csv nem található: {self.registry_path}")

        rows = []
        found = False

        with open(self.registry_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["experiment_name"] == self.experiment_name:
                    row["message"] = message
                    row["status"] = status
                    row["results"] = json.dumps(results, ensure_ascii=False)
                    row["args"] = json.dumps(vars(self.args), ensure_ascii=False)
                    row["run_path"] = self.path_name
                    found = True
                rows.append(row)

        if not found:
            raise RuntimeError(
                f"Nem található a frissítendő experiment bejegyzés a registry-ben: {self.experiment_name}"
            )

        with open(self.registry_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def save_results(self, results, message=""):
        self._save_results_files(results, message=message)
        self._update_registry_entry(results=results, message=message, status="finished")

    def save_failed_run(self, message="failed"):
        failed_results = {}
        self._save_results_files(failed_results, message=message)
        self._update_registry_entry(results=failed_results, message=message, status="failed")

    def get_path(self, folder_name=None, file_name=None):
        if folder_name is None:
            base_path = self.path_name
        elif folder_name == "forecast_examples":
            base_path = self.forecast_examples_path
        elif folder_name == "learning_curves":
            base_path = self.learning_curves_path
        else:
            raise ValueError(f"Ismeretlen mappa: {folder_name}")

        if file_name is None:
            return base_path

        return os.path.join(base_path, file_name)
"""