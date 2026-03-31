import os
import sys
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

        # Full paths
        self.task_path = os.path.join(self.drive_path, self.current_task_folder)
        self.path_name = os.path.join(self.task_path, self.current_time_folder)

        # Subfolders
        self.forecast_examples_path = os.path.join(self.path_name, "forecast_examples")
        self.learning_curves_path = os.path.join(self.path_name, "learning_curves")

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
        self._save_args_files()

        # create initial row in global csv
        self._create_registry_entry()


        print("---------------")
        print("EXPERIMENT PATH: ", self.path_name)
        print("---------------")

    def _save_args_files(self):
        args_txt_path = os.path.join(self.path_name, "args.txt")
        with open(args_txt_path, "w", encoding="utf-8") as f:
            for key, value in sorted(vars(self.args).items()):
                f.write(f"{key}: {value}\n")

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