import os
import numpy as np
import sys
import csv
import json
from datetime import datetime
from pathlib import Path

import gspread
from google.colab import auth
from google.auth import default

class Saver:
    def __init__(self, args): # full path example: /content/drive/MyDrive/itt_most/egyetem/onlab/run_results/forecasting_M_ETTh1/2026_03_21_16_51_21 and inside here: /forecast_examples and /learning_curves
        # Base folder
        self.drive_path = "/content/drive/MyDrive/itt_most/egyetem/onlab/run_results"

        # check if drive is mounted
        if not os.path.isdir(self.drive_path):
            raise FileNotFoundError(f"The given drive_path does not exist: {self.drive_path}")

        self.args = args
        self.args_dict = self._to_jsonable(vars(args))


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

        # Google Sheets registry config
        self.sheet_id = "1zcLxMAXpxRuy4ZeTHoJTOlm8nVL6P9zI6x6umotRNRk"
        self.worksheet_name = "registry"

        # main path
        #self.registry_path = os.path.join(self.drive_path, "run_registry.csv")


        # Create folders
        os.makedirs(self.task_path, exist_ok=True)
        try:
            os.makedirs(self.path_name, exist_ok=False)
        except FileExistsError:
            print(f"Folder already in use: {self.path_name}")
            sys.exit(1)
        os.makedirs(self.forecast_examples_path, exist_ok=True)
        os.makedirs(self.learning_curves_path, exist_ok=True)
        
        # init google sheets
        self.gc = self._init_google_sheets_client()
        self.registry_ws = self._get_or_create_registry_worksheet()

        # save args locally
        self._save_args_file()

        # create initial row in global csv
        self._create_registry_entry()


        print("---------------")
        print("EXPERIMENT PATH: ", self.path_name)
        print("---------------")

    def _init_google_sheets_client(self):
        auth.authenticate_user()
        creds, _ = default()
        return gspread.authorize(creds)

    def _get_or_create_registry_worksheet(self):
        sh = self.gc.open_by_key(self.sheet_id)

        try:
            ws = sh.worksheet(self.worksheet_name)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(
                title=self.worksheet_name,
                rows=1000,
                cols=max(10, len(self._registry_fieldnames()))
            )

        fieldnames = self._registry_fieldnames()
        current_header = ws.row_values(1)

        # Ensure header row exists and is correct
        if current_header != fieldnames:
            end_col = self._col_letter(len(fieldnames))
            ws.update(f"A1:{end_col}1", [fieldnames])

        return ws

    def _to_jsonable(self, obj):
        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, dict):
            return {key: self._to_jsonable(value) for key, value in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._to_jsonable(value) for value in obj]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.bool_):
            return bool(obj)

        if "torch" in str(type(obj)):
            if hasattr(obj, "detach"):
                obj = obj.detach().cpu()
            if hasattr(obj, "ndim"):
                if obj.ndim == 0:
                    return obj.item()
                return obj.tolist()

        return obj

    def _registry_fieldnames(self):
        return [
            "experiment_name",
            "message",
            "status",
            "results",
            "args",
            "run_path",
        ]

    def _col_letter(self, col_idx):
        result = ""
        while col_idx > 0:
            col_idx, rem = divmod(col_idx - 1, 26)
            result = chr(65 + rem) + result
        return result

    def _save_args_file(self):
        args_txt_path = os.path.join(self.path_name, "args.txt")
        with open(args_txt_path, "w", encoding="utf-8") as f:
            for key, value in sorted(self.args_dict.items()):
                f.write(f"{key}: {value}\n")

    def _build_registry_row(self, results="", message="", status="running"):
        return {
            "experiment_name": self.experiment_name,
            "message": message,
            "status": status,
            "results": json.dumps(self._to_jsonable(results), ensure_ascii=False),
            "args": json.dumps(self.args_dict, ensure_ascii=False),
            "run_path": self.path_name,
        }

    def _create_registry_entry(self):
        file_exists = os.path.isfile(self.registry_path)
        fieldnames = self._registry_fieldnames()

        row = {
            "experiment_name": self.experiment_name,
            "message": "",
            "status": "running",
            "results": "",
            "args": json.dumps(self.args_dict, ensure_ascii=False),
            "run_path": self.path_name,
        }

        with open(self.registry_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _find_experiment_row(self):
        try:
            cell = self.registry_ws.find(self.experiment_name, in_column=1)
            return cell.row
        except gspread.exceptions.CellNotFound:
            raise RuntimeError(
                f"Couldn't find experiment to update in registry: {self.experiment_name}"
            )
    
    def _save_results_files(self, results, message=""):
        results_json_path = os.path.join(self.path_name, "results.json")
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(self._to_jsonable(results), f, indent=2, ensure_ascii=False)

        message_path = os.path.join(self.path_name, "message.txt")
        with open(message_path, "w", encoding="utf-8") as f:
            f.write(message)

    def _update_registry_entry(self, results, message="", status="finished"):
        row_idx = self._find_experiment_row()

        fieldnames = self._registry_fieldnames()
        row = self._build_registry_row(results=results, message=message, status=status)
        values = [row[field] for field in fieldnames]

        end_col = self._col_letter(len(fieldnames))
        self.registry_ws.update(
            f"A{row_idx}:{end_col}{row_idx}",
            [values],
            value_input_option="RAW"
        )
        
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