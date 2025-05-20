import io
import json
import tempfile
from sklearn.model_selection import train_test_split
from google.cloud import storage
from dotenv import load_dotenv; load_dotenv()
import pandas as pd
from io import BytesIO

import numpy as np
from dotenv import load_dotenv
from google.cloud import storage
import os
import requests
import time
import pandas as pd
import re
import airtable_download_upload
from pyairtable import Api, Base, Table
from unidecode import unidecode
from datetime import datetime as dt
import time
import os
import shutil
from urllib.parse import quote

from geopy.geocoders import GoogleV3
from google.cloud import vision
from geopy.geocoders import Nominatim
import googlemaps
from collections import defaultdict
import logging
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import tensorflow_io as tfio
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
# Global configuration parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 30
TRAINING_INTERVAL_DAYS = 7
LEARNING_RATE = 1e-4


class GCPValidationClassification:
    def __init__(self, model_path=None):

        load_dotenv()
        self.GOOGLE_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GOOGLE_CREDENTIALS
        self.BUCKET_NAME = "building_images_storage"
        self.CONFIG_FILE_NAME = os.getenv("CONFIG_FILE_NAME")
        self.GOOGLE_MAPS_API_KEY = 'Google Maps Key'
        self.geolocator = Nominatim(user_agent="myGeolocator")
        self.gmaps = googlemaps.Client(key=self.GOOGLE_MAPS_API_KEY)
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.BUCKET_NAME)
        self.BASE_URL = 'https://api.airtable.com/v0/'
        self.api_key = 'Airtable API Key'
        self.base_name = 'Airtable Base Name'
        self.api = Api(self.api_key)
        self.model_path = model_path
        self.training_metadata_blob = "training_metadata.json"

        # Load the model immediately
        try:
            self.model = tf.keras.models.load_model(model_path)
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise

    def create_dic(self):
        """
        Creates two dictionaries:
        1. folder_files_fotos: Images inside the 'fotos_fincas' folder in GCP storage.
        2. folder_files_segmented: Images inside the 'segmented_fotos_fincas' folder in GCP storage.

        Returns:
            tuple: (folder_files_fotos, folder_files_segmented)
                - folder_files_fotos (dict): Keys are folders inside 'fotos_fincas',
                                            values are lists of image filenames.
                - folder_files_segmented (dict): Keys are folders inside 'segmented_fotos_fincas',
                                                values are lists of image filenames.
        """
        # Dictionary for fotos_fincas
        folder_files_fotos = defaultdict(list)
        print("Creating dictionary for fotos_fincas...")
        prefix_fotos = "fotos_fincas/"
        blobs_fotos = self.bucket.list_blobs(prefix=prefix_fotos)
        for blob in blobs_fotos:
            # Skip folder placeholders
            if blob.name.endswith('/'):
                print(f"Skipping folder: {blob.name}")
                continue
            folder = os.path.dirname(blob.name)
            print(f"Processing blob: {blob.name} in folder: {folder}")
            filename = os.path.basename(blob.name)
            print(f"Filename: {filename}")
            folder_files_fotos[folder].append(filename)
            print(f"Added {filename} to folder {folder}")   
        folder_files_fotos = dict(folder_files_fotos)
        print("Dictionary for fotos_fincas created successfully.")

        # Dictionary for segmented_fotos_fincas
        folder_files_segmented = defaultdict(list)
        print("Creating dictionary for segmented_fotos_fincas...")
        prefix_segmented = "segmented_fotos_fincas/"
        print(f"Looking for blobs in {prefix_segmented}")
        blobs_segmented = self.bucket.list_blobs(prefix=prefix_segmented)
        for blob in blobs_segmented:
            if blob.name.endswith('/'):
                print(f"Skipping folder: {blob.name}")
                continue
            folder = os.path.dirname(blob.name)
            print(f"Processing blob: {blob.name} in folder: {folder}")
            filename = os.path.basename(blob.name)
            print(f"Filename: {filename}")
            folder_files_segmented[folder].append(filename)
            print(f"Added {filename} to folder {folder}")
        folder_files_segmented = dict(folder_files_segmented)
        print("Dictionary for segmented_fotos_fincas created successfully.")


        return folder_files_fotos, folder_files_segmented

    def ascii_filename(self, file_name):
        # Convert non-ASCII characters to their closest ASCII counterparts
        name, extension = os.path.splitext(file_name)
        ascii_name = unidecode(name)
        # Replace spaces with underscores and remove non-alphanumeric characters
        ascii_name = ''.join(e for e in ascii_name if e.isalnum() or e in ('-', '_')).rstrip()
        return f"{ascii_name}{extension}"

    def validate_address(self, address, city):
        try:
            location = self.gmaps.addressvalidation([address],
                                                    regionCode='ES',
                                                    locality=city)['result']['address']['formattedAddress']
            return location

        except:
            return 'Error'

    def extract_postal_code(text):
        try:
            numbers = re.findall(r'\d+', text)
            large_numbers = [int(num) for num in numbers if int(num) > 1000]
            return str(int(large_numbers[0]))
        except:
            return None

    def download_df_fincas_nameproperly(self):
        print("Downloading the Fincas table from Airtable...")
        table = self.api.table(self.base_name, 'Fincas')
        columns_to_include = [
            ['FINCA', 'Año construcción', 'Tipo Finca', 'Parcela Catastral', 'parcela_catastral_joinkey',
             'Codigo Postal', 'Address Validated AI']]
        formula = "{parcela_catastral_joinkey}"
        table = table.all(view='testing', formula=formula, fields=columns_to_include)
        df_fincas_toSplit = pd.DataFrame(table)
        lista_records = list(df_fincas_toSplit['id'])
        df_fincas_toSplit = pd.DataFrame(list(df_fincas_toSplit['fields']))
        df_fincas_toSplit['record_id'] = lista_records
        df_fincas_toSplit['Codigo Postal'] = df_fincas_toSplit['Codigo Postal'].apply(
            lambda x: int(x) if pd.notna(x) else None).fillna('')
        df_fincas_toSplit['Finca_Proper_google'] = df_fincas_toSplit['FINCA'].str.title() + ', Madrid'
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['parcela_catastral_joinkey'].notna()]
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['Address Validated AI'].notna()]
        df_fincas_toSplit = df_fincas_toSplit[df_fincas_toSplit['Finca_Proper_google'].notna()]
        df_fincas_toSplit['ascii_filename'] = df_fincas_toSplit['Address Validated AI'].map(self.ascii_filename)
        print("Fincas table downloaded successfully.")
        dict_tipo_fincas = {'Representativa +5%': 'Clasica',
                            'Clásica +0%': 'Clasica',
                            'Moderna -10%': 'Moderna',
                            'Moderna-Clásica -5%': 'Moderna',
                            'Asintónica -20%': 'Moderna'}

        df_fincas_toSplit['Tipo Finca'] = df_fincas_toSplit['Tipo Finca'].replace(dict_tipo_fincas)
        print("Fincas table cleaned successfully.")
        return df_fincas_toSplit

    def finca_get_street_maps_photo_download(self, df, tipo, location):
        """
        Downloads the Street View and cadastral images for a given location and uploads
        them directly to the GCP bucket under the appropriate folder based on 'tipo'.

        The images are uploaded to:
            fotos_fincas/{tipo}/{parcela_catastral_joinkey}.jpg          (Street View image)
            fotos_fincas/{tipo}/{parcela_catastral_joinkey}_catastro.jpg   (Cadastral image)

        Args:
            df (pd.DataFrame): DataFrame containing the address and cadastral information.
            tipo (str): Classification type (should be one of "clasica", "noclasica", or "no_valorada").
            location (str): The key for the location (typically the parcela_catastral_joinkey).
            folder (str): (Unused in this version) Previously the local folder; now the bucket destination is fixed.
        """

        # Determine values from the DataFrame.
        parcela_catastral_joinkey = location
        if tipo == 'no_valorada':
            par_catastral = df[df['parcela_catastral_joinkey'] == location]['Parcela Catastral'].unique()[0]
            location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]
        else:
            par_catastral = df[df['parcela_catastral_joinkey'] == location]['Parcela Catastral'].unique()[0]
            location = df[df['parcela_catastral_joinkey'] == location]['Address Validated AI'].unique()[0]
        print(f"Got the {tipo} for location: {location}")

        # ---------------------
        # Download Street View Image
        # ---------------------
        meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
        pic_base = 'https://maps.googleapis.com/maps/api/streetview?'

        meta_params = {'key': self.GOOGLE_MAPS_API_KEY, 'location': location}
        pic_params = {
            'key': self.GOOGLE_MAPS_API_KEY,
            'location': location,
            'size': "640x640",
            'pitch': '30',
            'source': 'outdoor'
        }
        try:
            print(f"[DEBUG] Attempting to fetch Street View image for location: {location}")
            pic_response = requests.get(pic_base, params=pic_params)
            print(f"[DEBUG] Fetched Street View image. Status Code: {pic_response.status_code}")
        except Exception as e:
            print(f"[DEBUG] Error fetching Street View image: {e}. Retrying in 120 seconds.")
            time.sleep(120)
            try:
                print(f"[DEBUG] Retrying Street View image fetch for location: {location}")
                pic_response = requests.get(pic_base, params=pic_params)
                print(f"[DEBUG] Retry successful. Status Code: {pic_response.status_code}")
            except Exception as retry_e:
                print(f"[DEBUG] Retry failed: {retry_e}. Skipping this location.")
                return

        # Upload the Street View image directly to GCS.
        try:
            # Construct the blob path:
            # e.g. "fotos_fincas/clasica/<parcela_catastral_joinkey>.jpg"
            street_view_blob_path = f"fotos_fincas/{tipo}/{parcela_catastral_joinkey}.jpg"
            self.bucket.blob(street_view_blob_path).upload_from_string(
                pic_response.content, content_type='image/jpeg'
            )
            print(f"[DEBUG] Uploaded Street View image to {street_view_blob_path}")
        except Exception as e:
            print(f"[DEBUG] Error uploading Street View image to GCS: {e}. Skipping this location.")
        pic_response.close()

        # ---------------------
        # Download and Upload Cadastral Image
        # ---------------------
        if type(par_catastral) is not float:
            cadastral_url = (
                    'http://ovc.catastro.meh.es/OVCServWeb/OVCWcfLibres/OVCFotoFachada.svc/'
                    'RecuperarFotoFachadaGet?ReferenciaCatastral=' + par_catastral
            )
            print(f"[DEBUG] Constructed cadastral image URL: {cadastral_url}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/58.0.3029.110 Safari/537.3'
            }
            print(f"[DEBUG] Using headers for cadastral image request: {headers}")
            print("[DEBUG] Sending GET request for cadastral image.")
            try:
                cadastral_response = requests.get(cadastral_url, headers=headers)
                print(f"[DEBUG] Received cadastral image. Status Code: {cadastral_response.status_code}")
            except Exception as e:
                print(f"[DEBUG] Exception during GET for cadastral image: {e}")
                raise

            cadastral_blob_path = f"fotos_fincas/{tipo}/{parcela_catastral_joinkey}catastro.jpg"
            self.bucket.blob(cadastral_blob_path).upload_from_string(
                cadastral_response.content, content_type='image/jpeg'
            )
            print(f"[DEBUG] Uploaded cadastral image to {cadastral_blob_path}")

    def check_image_existence(self, folder_files, parcela_catastral_joinkey):
        print(f"[DEBUG] Checking existence of images for parcela_catastral_joinkey: {parcela_catastral_joinkey}")

        for folder, files in folder_files.items():
            print(f"[DEBUG] Inspecting folder: {folder}")

            expected_images = [
                f"{parcela_catastral_joinkey}.jpg",
                f"{parcela_catastral_joinkey}catastro.jpg"
                f"{parcela_catastral_joinkey}_catastro.jpg"
            ]
            print(f"[DEBUG] Expected images: {expected_images}")

            for img in expected_images:
                print(f"[DEBUG] Checking for image: {img} in folder: {folder}")
                if img in files:
                    print(f"[DEBUG] Image {img} found in folder {folder}")
                    return True, folder
                else:
                    print(f"[DEBUG] Image {img} not found in folder {folder}")

        print(f"[DEBUG] Images for {parcela_catastral_joinkey} are missing in all folders")
        return False, None

    def check_images_for_sampled_data(self, folder_files, sampled_data):
        print("[DEBUG] Starting check_images_for_sampled_data")
        not_found_records = []

        for joinkey in sampled_data['parcela_catastral_joinkey']:
            print(f"[DEBUG] Checking images for joinkey: {joinkey}")
            result, folder = self.check_image_existence(folder_files, joinkey)
            if not result:
                print(f"[DEBUG] Images for joinkey {joinkey} not found")
                not_found_records.append(joinkey)
            else:
                print(f"[DEBUG] Images for joinkey {joinkey} found in folder {folder}")
            print(f"[DEBUG] Checked {joinkey}: {'Found in ' + folder if result else 'Not Found'}")

        # Create a DataFrame for not found records
        print("[DEBUG] Creating DataFrame for not found records")
        not_found_df = sampled_data[sampled_data['parcela_catastral_joinkey'].isin(not_found_records)]
        print(f"[DEBUG] Total missing records: {len(not_found_records)}")
        return not_found_df

    def download_missing_images(self, missing_images_df, df):
        print("[DEBUG] Starting download_missing_images")

        for _, row in missing_images_df.iterrows():
            joinkey = row['parcela_catastral_joinkey']
            tipo_finca = str(row['Tipo Finca']).strip().lower()
            print(f"[DEBUG] Processing joinkey: {joinkey}, Tipo Finca: {tipo_finca}")

            if tipo_finca in ['', 'nan', 'none']:
                tipo = 'no_valorada'
                print(f"[DEBUG] Tipo Finca is empty or invalid. Set tipo to 'no_valorada'")
            elif tipo_finca == 'clasica':
                tipo = 'clasica'
                print(f"[DEBUG] Tipo Finca is 'clasica'. Set tipo to 'clasica'")
            else:
                tipo = 'noclasica'
                print(f"[DEBUG] Tipo Finca is '{tipo_finca}'. Set tipo to 'noclasica'")

            try:
                print(f"[DEBUG] Calling finca_get_street_maps_photo_download for joinkey: {joinkey}")
                self.finca_get_street_maps_photo_download(df, tipo, joinkey)
                print(f"[DEBUG] Successfully called finca_get_street_maps_photo_download for joinkey: {joinkey}")
            except FileNotFoundError as e:
                print(f"Error saving image: {e}")
                print(f"[DEBUG] FileNotFoundError encountered for joinkey: {joinkey}. Continuing to next record.")
                continue
            except Exception as e:
                print(f"[DEBUG] Unexpected error while downloading image for {joinkey}: {e}")
                continue

    def segment_image_array(self, image: np.ndarray, model, target_label: str = "main_building",
                            threshold: float = 0.80) -> np.ndarray:
        """
        Runs YOLO prediction on the input image and extracts the best bounding box (with label and confidence)
        for the target label ("main_building") if its confidence is at least the threshold.
        Otherwise, crops the image from the center by removing 15% margins.
        This version works entirely in-memory.
        
        If a valid detection is found, the function crops the image using the detected bounding box.
        """
        # Convert image to RGB for prediction.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        try:
            results = model.predict(source=image_rgb, save=False, imgsz=640, iou=0.5)
        except Exception as e:
            logging.error(f"Error during YOLO prediction: {e}")
            return image_rgb

        best_confidence = 0
        best_box = None
        any_detection = False

        # Loop through all predictions and track the best detection for the target label.
        for result in results:
            for box, confidence, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label_name = model.names[int(cls)]
                if label_name == target_label:
                    any_detection = True
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_box = box.cpu().numpy().astype(int)

        if any_detection and best_box is not None and best_confidence >= threshold:
            x1, y1, x2, y2 = best_box
            height, width, _ = image_rgb.shape
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                logging.warning("Invalid bounding box; will crop center region instead.")
            else:
                logging.info(f"Detection found: {best_box} with confidence {best_confidence:.2f}")
                # Crop the image using the detected bounding box
                cropped = image_rgb[y1:y2, x1:x2]
                return cropped

        # If no valid detection is found, crop the center of the image by removing 15% margins.
        logging.info("No valid detection with confidence >= 0.80; cropping center region.")
        height, width, _ = image_rgb.shape
        x1 = int(0.15 * width)
        y1 = int(0.15 * height)
        x2 = int(0.85 * width)
        y2 = int(0.85 * height)
        cropped_center = image_rgb[y1:y2, x1:x2]
        return cropped_center

    def segment_and_upload_directories(self, input_dirs: list, model, folder_files_segmented: dict, force_resegment: bool = False) -> None:
        """
        Processes images in the specified GCS directories, applies YOLO segmentation (using the logic above),
        and uploads the processed images (annotated or center-cropped) to GCS.

        The output structure will be:
        - Input:  gs://building_images_storage/fotos_fincas/clasica/
                Output: gs://building_images_storage/segmented_fotos_fincas/clasica/
        - Input:  gs://building_images_storage/fotos_fincas/noclasica/
                Output: gs://building_images_storage/segmented_fotos_fincas/noclasica/
        - Input:  gs://building_images_storage/fotos_fincas/no_valorada/
                Output: gs://building_images_storage/segmented_fotos_fincas/no_valorada/


        Args:
            input_dirs (list): List of GCS directory URIs containing image files.
            model (YOLO): Loaded YOLO model for segmentation.
            folder_files_segmented (dict): Dictionary of segmented images (keys are folder names under segmented_fotos_fincas)
            force_resegment (bool): If True, segment all images regardless of whether they are already segmented.
        """
        allowed_extensions = (".jpg", ".jpeg", ".png")
        all_output_blobs = []  # Collect output blob URIs.

        for input_dir in input_dirs:
            if not input_dir.startswith("gs://"):
                logging.error(f"Invalid GCS URI: {input_dir}")
                continue

            # Parse the folder name (e.g., "clasica", "noclasica", "no_valorada").
            parts = input_dir.split('/')
            if len(parts) < 4:
                logging.error(f"Could not parse bucket and folder from URI: {input_dir}")
                print(f"Could not parse bucket and folder from URI: {input_dir}")
                continue
            folder_name = parts[-2]
            logging.info(f"Processing directory: {input_dir} (folder: {folder_name})")
            print(f"Processing directory: {input_dir} (folder: {folder_name})")

            # Define the output prefix.
            output_prefix = f"segmented_fotos_fincas/{folder_name}/"
            # For checking segmented images, determine the key in the segmented dictionary.
            segmented_key = output_prefix.rstrip('/')  # e.g., "segmented_fotos_fincas/clasica"
            print(f"Segmented key: {segmented_key}")

            # Remove the bucket prefix to get the blob prefix.
            input_prefix = input_dir.replace(f"gs://{self.BUCKET_NAME}/", "")
            blobs = self.bucket.list_blobs(prefix=input_prefix)
            count = 0

            for blob in blobs:

                if not blob.name.lower().endswith(allowed_extensions):
                    continue

                filename = blob.name.split('/')[-1]
                # Check if this image is already segmented (only if force_resegment is False)
                if not force_resegment:
                    if segmented_key in folder_files_segmented and filename in folder_files_segmented[segmented_key]:
                        logging.info(f"Skipping {blob.name} as it is already segmented.")
                        print(f"Skipping {blob.name} as it is already segmented.")
                        continue

                logging.info(f"Processing blob: {blob.name}")
                print(f"Processing blob: {blob.name}")
                try:
                    image_bytes = blob.download_as_bytes()
                except Exception as e:
                    logging.error(f"Error downloading blob {blob.name}: {e}")
                    print(f"Error downloading blob {blob.name}: {e}")
                    continue

                nparr = np.frombuffer(image_bytes, np.uint8)
                try:
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception as e:
                    logging.error(f"Error decoding image from blob: {blob.name}")
                    print(f"Error decoding image from blob: {blob.name}")
                    continue
                if image is None:
                    logging.error(f"Failed to decode image from blob: {blob.name}")
                    print(f"Failed to decode image from blob: {blob.name}")
                    continue

                # Process the image: annotate with detection (if available) or crop center.
                print(f"Segmenting image: {blob.name}")
                processed_image = self.segment_image_array(image, model, target_label="main_building", threshold=0.80)
                print(f"Image segmented: {blob.name}")

                # Convert processed image from RGB to BGR for JPEG encoding.
                processed_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                success, buffer = cv2.imencode('.jpg', processed_bgr)
                if not success:
                    logging.error(f"Failed to encode processed image for blob: {blob.name}")
                    continue
                processed_bytes = buffer.tobytes()

                # Determine output file name.
                output_blob_name = f"{output_prefix}{filename}"
                try:
                    out_blob = self.bucket.blob(output_blob_name)
                    out_blob.upload_from_string(processed_bytes, content_type='image/jpeg')
                    full_output_uri = f"gs://{self.BUCKET_NAME}/{output_blob_name}"
                    logging.info(f"Uploaded processed image to: {full_output_uri}")
                    all_output_blobs.append(full_output_uri)
                except Exception as e:
                    logging.error(f"Error uploading processed image to {output_blob_name}: {e}")
                    continue

        # Print all output blob paths.
        print("\nProcessed images uploaded to the following GCS paths:")
        for output_uri in all_output_blobs:
            print(output_uri)


    def list_blobs_in_prefix(self, prefix: str, allowed_extensions=(".jpg", ".jpeg", ".png")):
        """
        Lists all blobs in the bucket with the given prefix, filtering by allowed image extensions.
        """
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        filtered = [blob for blob in blobs if blob.name.lower().endswith(allowed_extensions)]
        return filtered

    def copy_blobs(self, blob_list, output_prefix: str):
        """
        Copies each blob in blob_list to the destination under output_prefix within the same bucket.
        Returns a list of destination URIs.
        """
        output_uris = []
        for blob in blob_list:
            filename = blob.name.split('/')[-1]
            dest_blob_name = f"{output_prefix}{filename}"
            self.bucket.copy_blob(blob, self.bucket, dest_blob_name)
            output_uri = f"gs://{self.BUCKET_NAME}/{dest_blob_name}"
            output_uris.append(output_uri)
            logging.info(f"Copied {blob.name} to {dest_blob_name}")
        return output_uris

    def copy_blob_with_retries(self, blob, dest_blob_name, max_retries=3):
        """
        Attempts to copy a blob with retries in case of connection errors.
        """
        for attempt in range(max_retries):
            try:
                self.bucket.copy_blob(blob, self.bucket, dest_blob_name)
                print(f"Successfully copied {blob.name} to {dest_blob_name}")
                return True
            except requests.exceptions.ConnectionError as e:
                print(f"Attempt {attempt + 1} failed for blob {blob.name}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"Max retries reached for blob {blob.name}.")
                    raise

    def copy_blobs(self, blobs, output_prefix):
        """
        Copies a list of blobs to the given output prefix using retry logic.
        Returns a list of output URIs for the successfully copied blobs.
        """
        output_uris = []
        for blob in blobs:
            # Create destination blob name using the output prefix and original filename.
            dest_blob_name = f"{output_prefix}{blob.name.split('/')[-1]}"
            try:
                self.copy_blob_with_retries(blob, dest_blob_name)
                # Construct the full gs:// URI for the copied blob.
                output_uri = f"gs://{self.bucket.name}/{dest_blob_name}"
                output_uris.append(output_uri)
            except Exception as e:
                print(f"Failed to copy {blob.name} after retries: {e}")
        return output_uris

    def train_val_split_gcs(self):
        """
        Splits images from the segmented folders ("segmented_fotos_fincas/clasica/" and
        "segmented_fotos_fincas/noclasica/") into training and validation sets,
        and copies them into the following output structure:

            gs://building_images_storage/training_fotos_fincas/train/clasica/
            gs://building_images_storage/training_fotos_fincas/validation/clasica/
            gs://building_images_storage/training_fotos_fincas/train/noclasica/
            gs://building_images_storage/training_fotos_fincas/validation/noclasica/

        The split is done with 70% of images for training and 30% for validation.
        """
        # Define source prefixes.
        print("\nStarting training and validation split process...")
        source_clasica = "segmented_fotos_fincas/clasica/"
        source_noclasica = "segmented_fotos_fincas/noclasica/"
        print(f"Source Clasica: {source_clasica}")
        print(f"Source Noclasica: {source_noclasica}")

        # List blobs from each category.
        clasica_blobs = self.list_blobs_in_prefix(source_clasica)
        print(f"Found {len(clasica_blobs)} clasica images.")
        noclasica_blobs = self.list_blobs_in_prefix(source_noclasica)
        print(f"Found {len(noclasica_blobs)} noclasica images.")
        logging.info(f"Found {len(clasica_blobs)} clasica images and {len(noclasica_blobs)} noclasica images.")
        print(f"Found {len(clasica_blobs)} clasica images and {len(noclasica_blobs)} noclasica images.")

        # Split each category into train (70%) and validation (30%).
        train_clasica, val_clasica = train_test_split(clasica_blobs, test_size=0.3, random_state=42)
        train_noclasica, val_noclasica = train_test_split(noclasica_blobs, test_size=0.3, random_state=42)

        # Define output prefixes.
        output_prefixes = {
            "train_clasica": "training_fotos_fincas/train/clasica/",
            "val_clasica": "training_fotos_fincas/validation/clasica/",
            "train_noclasica": "training_fotos_fincas/train/noclasica/",
            "val_noclasica": "training_fotos_fincas/validation/noclasica/"
        }

        # Copy blobs to the respective output locations.
        print("\nCopying blobs to the following output locations:")
        out_train_clasica = self.copy_blobs(train_clasica, output_prefixes["train_clasica"])
        print("\nCopied Clasica training images.")
        out_val_clasica = self.copy_blobs(val_clasica, output_prefixes["val_clasica"])
        print("\nCopied Clasica validation images.")
        out_train_noclasica = self.copy_blobs(train_noclasica, output_prefixes["train_noclasica"])
        print("\nCopied Noclasica training images.")
        out_val_noclasica = self.copy_blobs(val_noclasica, output_prefixes["val_noclasica"])
        print("\nCopied Noclasica validation images.")

        # Print output URIs.
        print("\nTraining Clasica:")
        for uri in out_train_clasica:
            print(uri)
        print("\nValidation Clasica:")
        for uri in out_val_clasica:
            print(uri)
        print("\nTraining Noclasica:")
        for uri in out_train_noclasica:
            print(uri)
        print("\nValidation Noclasica:")
        for uri in out_val_noclasica:
            print(uri)

    def get_last_training_date(self):
        """Get last training date from GCS"""
        try:
            blob = self.bucket.blob(self.training_metadata_blob)
            if blob.exists():
                content = blob.download_as_string()
                data = json.loads(content)
                if data['last_training']:  # Check if not None
                    # Parse the ISO format string directly
                    try:
                        date_str = data['last_training']
                        return dt.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        # Try without microseconds
                        return dt.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
            return None
        except Exception as e:
            logging.warning(f"Error reading last training date: {e}")
            return None

    def should_train(self):
        """Check if enough time has passed since last training"""
        last_training = self.get_last_training_date()
        if last_training is None:
            logging.info("No previous training date found. Will proceed with training.")
            return True

        current_time = dt.now()
        days_since_training = (current_time - last_training).days

        if days_since_training < TRAINING_INTERVAL_DAYS:
            logging.info(f"Only {days_since_training} days since last training. "
                         f"Waiting for {TRAINING_INTERVAL_DAYS - days_since_training} more days.")
            return False

        logging.info(f"{days_since_training} days since last training. Will proceed with training.")
        return True

    def update_training_date(self):
        """Update the last training date in GCS"""
        try:
            metadata = {
                'last_training': dt.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
                'model_path': self.model_path
            }
            blob = self.bucket.blob(self.training_metadata_blob)
            blob.upload_from_string(json.dumps(metadata))
            logging.info("Updated last training date in GCS")
        except Exception as e:
            logging.error(f"Failed to update training date in GCS: {e}")

    def get_file_list_from_gcs(self, directory, allowed_extensions=(".jpg", ".jpeg", ".png")):
        prefix = directory.replace(f"gs://{self.BUCKET_NAME}/", "")
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        file_list = [f"gs://{self.BUCKET_NAME}/{blob.name}"
                     for blob in blobs if blob.name.lower().endswith(allowed_extensions)]
        return file_list

    def get_dataset(self, directory, img_size):
        file_list = self.get_file_list_from_gcs(directory)
        if not file_list:
            raise FileNotFoundError(f"Directory {directory} not found or is empty.")
        logging.info(f"Found {len(file_list)} files in {directory}.")

        classes = sorted(list({os.path.basename(os.path.dirname(path)) for path in file_list}))
        logging.info(f"Inferred classes: {classes}")

        labels = [classes.index(os.path.basename(os.path.dirname(path))) for path in file_list]
        dataset = tf.data.Dataset.from_tensor_slices((file_list, labels))

        def load_and_preprocess(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, img_size)
            image = image / 255.0
            label = tf.one_hot(label, depth=len(classes))
            return image, label

        return dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    def retrain(self, train_dir, val_dir):
        # """Retrain the model if enough time has passed"""
        if not self.should_train():
            return None
        
        # Clear TensorFlow session to avoid memory issues
        tf.keras.backend.clear_session()

        logging.info("Starting model retraining...")
        MODEL_BUCKET_NAME = "sophiq_static_files"
        GCS_MODEL_PATH = "models/fine_tuned_classification_model.h5"
        model_bucket = self.storage_client.bucket(MODEL_BUCKET_NAME)
        
        # ✅ Augmentation Pipeline
        train_transforms = A.Compose([
            A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5)], p=0.5),
            A.GridDropout(ratio=0.2, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize()
        ])

        val_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize()
        ])

        # ✅ Function to Load Image Paths from GCS - FIXED to handle subdirectories
        def get_file_list_from_gcs(directory, allowed_extensions=(".jpg", ".jpeg", ".png")):
            # Strip the gs:// prefix if present
            if directory.startswith("gs://"):
                prefix = directory.replace(f"gs://{self.BUCKET_NAME}/", "")
            else:
                prefix = directory
            
            # List all blobs including those in subdirectories
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            file_list = []
            
            for blob in blobs:
                if blob.name.lower().endswith(allowed_extensions) and not blob.name.endswith('/'):
                    file_list.append(f"gs://{self.BUCKET_NAME}/{blob.name}")
            
            if not file_list:
                print(f"❌ No images found in {directory} or its subdirectories. Check GCS path.")
            else:
                print(f"✅ Found {len(file_list)} images in {directory} and its subdirectories")
            
            return file_list

        # ✅ Extract Class Names - hard-coded to ensure correct ordering
        def extract_classes(file_list):
            # We're hard-coding the classes in the correct order: clasica=0, noclasica=1
            return ['clasica', 'noclasica']

        # ✅ Function to Assign Labels with hard-coded class mapping - FIXED to handle subdirectory structure
        def assign_labels(file_list, class_names):
            """Assigns labels based on hard-coded mapping: 'clasica'=0, 'noclasica'=1"""
            labels = []
            for path in file_list:
                try:
                    # Look for clasica or noclasica in the path
                    if '/clasica/' in path:
                        labels.append(0)  # Hard-coded 0 for 'clasica'
                    elif '/noclasica/' in path:
                        labels.append(1)  # Hard-coded 1 for 'noclasica'
                    else:
                        print(f"⚠️ Warning: Could not identify class in path {path}! Assigning -1.")
                        labels.append(-1)
                except Exception as e:
                    print(f"⚠️ Error extracting class from path {path}: {e}")
                    labels.append(-1)
            return labels

        # ✅ Get File Lists for Training & Validation
        train_files = get_file_list_from_gcs(train_dir)
        val_files = get_file_list_from_gcs(val_dir)

        # ✅ Hard-code class names in the correct order
        class_names = ['clasica', 'noclasica']
        print(f"🏷️ Class Names (hard-coded order): {class_names}")

        # ✅ Assign Labels with hard-coded mapping
        train_labels = assign_labels(train_files, class_names)
        val_labels = assign_labels(val_files, class_names)

        # ✅ Count images per class
        train_class_counts = {}
        val_class_counts = {}

        for class_name in class_names:
            train_class_counts[class_name] = len([f for f in train_files if f'/{class_name}/' in f])
            val_class_counts[class_name] = len([f for f in val_files if f'/{class_name}/' in f])

        print(f"🏷️ Class Names: {class_names}")
        print(f"🔖 Training Labels Sample: {train_labels[:10]}")
        print(f"📊 Training images per class: {train_class_counts}")
        print(f"📊 Validation images per class: {val_class_counts}")

        # ✅ Check if we have enough images to train
        if train_class_counts['clasica'] == 0 or train_class_counts['noclasica'] == 0:
            logging.error("Missing training data for one or more classes! Aborting training.")
            return None

        # ✅ Modified Function to Load & Preprocess Image from GCS
        def load_and_preprocess_gcs_image(file_path, label):
            """Load image from GCS, apply transformations, and return tensors with fixed shapes"""
            try:
                # Read the file
                image = tf.io.read_file(file_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, IMG_SIZE)
                
                # Convert to NumPy for albumentations
                image_np = image.numpy().astype(np.uint8)
                
                # Apply transformations based on whether it's training or validation
                # Here we're using train_transforms for all - in production you'd differentiate
                augmented = train_transforms(image=image_np)
                image_np = augmented["image"]
                
                # Convert back to tensor
                image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
                label_tensor = tf.convert_to_tensor(label, dtype=tf.int32)
                
                return image_tensor, label_tensor
            except Exception as e:
                print(f"⚠️ Error processing image {file_path}: {e}")
                # Return a placeholder image and label on error
                return tf.zeros((224, 224, 3), dtype=tf.float32), tf.constant(-1, dtype=tf.int32)

        # ✅ Function to set shapes after py_function
        def set_shapes(image, label):
            image.set_shape([224, 224, 3])
            label.set_shape([])
            return image, label

        # ✅ Modified Convert to TensorFlow Datasets with explicit shapes and shuffling
        train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_files, val_labels))

        # ✅ Apply preprocessing with explicit shape setting and proper shuffling
        train_dataset = (train_dataset
                        .shuffle(buffer_size=len(train_files), reshuffle_each_iteration=True)  # Add shuffling
                        .map(lambda x, y: tf.py_function(
                            func=load_and_preprocess_gcs_image, 
                            inp=[x, y], 
                            Tout=[tf.float32, tf.int32]), 
                            num_parallel_calls=tf.data.AUTOTUNE)
                        .map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(BATCH_SIZE)
                        .prefetch(tf.data.AUTOTUNE))

        val_dataset = (val_dataset
                    .map(lambda x, y: tf.py_function(
                        func=load_and_preprocess_gcs_image, 
                        inp=[x, y], 
                        Tout=[tf.float32, tf.int32]), 
                        num_parallel_calls=tf.data.AUTOTUNE)
                    .map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))

        # ✅ Set up fine-tuning - FIXED to properly handle existing model without stacking
        try:
            # Make only specific layers trainable for fine-tuning
            # We'll look for Dense and Dropout layers (usually near the end of the network)
            fine_tune_lr = LEARNING_RATE / 10  # Use a lower learning rate for fine-tuning
            
            # First, make all layers non-trainable
            for layer in self.model.layers:
                layer.trainable = False
                
            # Find the layers near the end (usually Dense, Dropout, etc. before the output)
            # These are the ones we want to fine-tune
            for layer in self.model.layers[-3:]:  # Last few layers
                if isinstance(layer, layers.Dense) or isinstance(layer, layers.Dropout):
                    print(f"Making layer trainable: {layer.name}")
                    layer.trainable = True
                    
            # Recompile the model with a lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
                weighted_metrics=['accuracy']
            )
            
            print("Model prepared for fine-tuning")
        except Exception as e:
            logging.error(f"Error setting up model for fine-tuning: {e}")
            return None

        # Add batch monitoring class to track class distribution during training
        class BatchMonitor(tf.keras.callbacks.Callback):
            def __init__(self, monitor_batches=10):
                super(BatchMonitor, self).__init__()
                self.monitor_batches = monitor_batches
                self.batches_seen = 0
            
            def on_train_batch_end(self, batch, logs=None):
                if self.batches_seen < self.monitor_batches:
                    # Process the most recent batch from logs
                    print(f"Batch {batch+1} completed - loss: {logs.get('loss', 'N/A'):.4f}, accuracy: {logs.get('accuracy', 'N/A'):.4f}")
                    self.batches_seen += 1
            
            def on_epoch_begin(self, epoch, logs=None):
                print(f"\n🔍 Monitoring first {self.monitor_batches} batches of epoch {epoch+1}...")
                self.batches_seen = 0

        # ✅ Callbacks
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
        log_dir = "logs/fit/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        batch_monitor = BatchMonitor(monitor_batches=5)  # Monitor first 5 batches of each epoch

        # ✅ Train Model with class weights and monitoring
        try:
            # Print batch info before training starts
            print("\n🔍 Examining distribution in first few batches to verify both classes are present:")
            batch_count = 0
            for images, labels in train_dataset.take(3):  # Check first 3 batches
                batch_count += 1
                unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
                distribution = {class_names[int(label)]: count for label, count in zip(unique_labels, counts)}
                print(f"  Batch {batch_count} distribution: {distribution}")

            class_weights = {
                0: 1.0,  # reference class (clasica)
                1: train_class_counts['clasica'] / train_class_counts['noclasica'] if train_class_counts['noclasica'] > 0 else 1.7
            }
            # Start the actual training
            history = self.model.fit(
                train_dataset,
                epochs=FINE_TUNE_EPOCHS,
                validation_data=val_dataset,
                class_weight=class_weights,
                callbacks=[lr_reduction, tensorboard_callback, batch_monitor]
            )
            self.model.summary()
            print("✅ Model training completed.")
            # ✅ Save Model to GCS
            print("Saving model to GCS...")
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
                temp_path = temp_file.name
            self.model.save(temp_path)

            # ✅ Upload Fine-Tuned Model to GCS
            blob = model_bucket.blob(GCS_MODEL_PATH)
            with open(temp_path, "rb") as f:
                model_buffer = io.BytesIO(f.read())
            model_buffer.seek(0)
            blob.upload_from_file(model_buffer, content_type="application/x-hdf5")

            print(f"✅ Fine-tuned model saved to GCS: gs://{MODEL_BUCKET_NAME}/{GCS_MODEL_PATH}")
            self.model_path = f"gs://{MODEL_BUCKET_NAME}/{GCS_MODEL_PATH}"
            
        except Exception as e:
            print(f"❌ Error during model training: {e}")
            return None

        # Update the last training date
        self.update_training_date()
        print("✅ Training date updated in GCS.")
        history = None
        return self.model_path, history
    def run_dict_fotos_fincas(self, df_fincas_toSplit):
        # Define the GCS directories to search.
        directories = [
            "gs://building_images_storage/segmented_fotos_fincas/clasica/",
            "gs://building_images_storage/segmented_fotos_fincas/noclasica/",
            "gs://building_images_storage/segmented_fotos_fincas/no_valorada/"
        ]

        # Build a dictionary to hold the filenames available in each GCS directory.
        dir_files = {}
        for directory in directories:
            # Remove the bucket prefix to get the internal prefix.
            prefix = directory.replace(f"gs://{self.BUCKET_NAME}/", "")
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            # Collect just the filename (the last part of the blob name) for each blob.
            files_set = set(blob.name.split("/")[-1] for blob in blobs if not blob.name.endswith("/"))
            dir_files[directory] = files_set

        dict_fotos_fincas = {}

        # Loop through each unique property key.
        for parcela_catastral_joinkey in df_fincas_toSplit.parcela_catastral_joinkey.unique():
            # Get the corresponding record_id from the DataFrame.
            record_id = df_fincas_toSplit[df_fincas_toSplit['parcela_catastral_joinkey'] == parcela_catastral_joinkey][
                'record_id'].unique()[0]

            # List to store found file paths.
            list_files_fincas = []

            # Prepare expected filenames.
            expected_filenames = [
                parcela_catastral_joinkey + 'catastro.jpg',
                parcela_catastral_joinkey + '.jpg'
            ]

            # Loop through each directory and check if any expected filename exists.
            for directory in directories:
                for fname in expected_filenames:
                    if fname in dir_files[directory]:
                        # Append the full GCS URI for the file.
                        list_files_fincas.append(f"{directory}{fname}")

            dict_fotos_fincas[record_id] = list_files_fincas

        return dict_fotos_fincas

    def generate_df_streetmaps_catastro_paths(self, dict_fotos_fincas):
        df_fincas_valoradas = pd.DataFrame()

        record_id_list = []
        paths_list = []
        catastro_list = []
        streetmaps_list = []
        i = 0

        for record_id, paths in dict_fotos_fincas.items():
            i += 1

            record_id_list.append(record_id)
            paths_list.append(paths)

        df_fincas_valoradas['record_id'] = record_id_list
        df_fincas_valoradas['paths'] = paths_list

        df_fincas_valoradas[['catastro_path', 'streetmaps_path']] = df_fincas_valoradas['paths'].apply(
            self.separate_paths)
        return df_fincas_valoradas

    def separate_paths(self, paths):
        catastro_path = None
        other_path = None

        for path in paths:
            if path.endswith("catastro.jpeg") or path.endswith("catastro.jpg"):
                catastro_path = path
            else:
                other_path = path

        return pd.Series([catastro_path, other_path])

    def imread_unicode(self, filename, flags=cv2.IMREAD_COLOR):
        """
        Reads an image from a file using a Unicode-friendly approach.
        This function reads the file as a byte array and decodes it with OpenCV.

        Args:
            filename (str): The path to the image file.
            flags (int): Flags for cv2.imdecode (default is cv2.IMREAD_COLOR).

        Returns:
            np.ndarray or None: The loaded image, or None if loading fails.
        """
        try:
            if not os.path.exists(filename):
                logging.error(f"File does not exist: {filename}")
                return None
            # Read file content as bytes
            img_array = np.fromfile(filename, np.uint8)
            img = cv2.imdecode(img_array, flags)
            return img
        except Exception as e:
            logging.error(f"Error reading image with Unicode support: {e}")
            return None

    def predict_image_from_array(self, image_array, img_height, img_width, model, class_names):
        """
        Processes an image array (e.g., a segmented image), resizes it to the required dimensions,
        and applies the classification model. It returns the predicted class name and the associated confidence.

        Args:
            image_array (np.ndarray): The image array (segmented image).
            img_height (int): Target image height.
            img_width (int): Target image width.
            model (tf.keras.Model): The trained classification model.
            class_names (list): List of class names corresponding to model outputs.

        Returns:
            tuple: (predicted_class, confidence)
        """
        # Resize the image to match the model's input shape.
        resized_image = cv2.resize(image_array, (img_width, img_height))
        # Normalize pixel values to [0, 1] (adjust preprocessing as required by your model)
        image = resized_image.astype('float32') / 255.0
        # Expand dimensions to create a batch of 1
        image = np.expand_dims(image, axis=0)
        # Get model predictions
        predictions = model.predict(image)
        # Get the class index with the highest probability
        pred_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][pred_idx]
        predicted_class = class_names[pred_idx]
        return predicted_class, confidence

    def apply_model_generate_df_classification_confidence(self, df_fincas_valoradas, model, img_height, img_width,
                                                          class_names):
        """
        For each row in the provided DataFrame, loads images from GCS paths and applies the classification model
        """
        catastro_classification_list = []
        catastro_confidence_list = []
        streetmaps_classification_list = []
        streetmaps_confidence_list = []

        # Iterate over each row in the DataFrame
        for _, row in df_fincas_valoradas.iterrows():
                # Process catastro image if path is available
                if row.catastro_path and row.catastro_path != '':
                    try:
                        # Download image from GCS
                        blob = self.bucket.blob(row.catastro_path.replace(f"gs://{self.BUCKET_NAME}/", ""))
                        image_bytes = blob.download_as_bytes()
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if image is not None:
                            classification, confidence = self.predict_image_from_array(image, img_height, img_width, model,
                                                                                    class_names)
                            catastro_classification_list.append(classification)
                            catastro_confidence_list.append(confidence)
                        else:
                            catastro_classification_list.append(np.NaN)
                            catastro_confidence_list.append(np.NaN)
                    except Exception as e:
                        logging.error(f"Error processing catastro image: {e}")
                        catastro_classification_list.append(np.NaN)
                        catastro_confidence_list.append(np.NaN)
                else:
                    catastro_classification_list.append(np.NaN)
                    catastro_confidence_list.append(np.NaN)

                # Process streetmaps image if path is available
                if row.streetmaps_path and row.streetmaps_path != '':
                    try:
                        # Download image from GCS
                        blob = self.bucket.blob(row.streetmaps_path.replace(f"gs://{self.BUCKET_NAME}/", ""))
                        image_bytes = blob.download_as_bytes()
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if image is not None:
                            classification, confidence = self.predict_image_from_array(image, img_height, img_width, model,
                                                                                    class_names)
                            streetmaps_classification_list.append(classification)
                            streetmaps_confidence_list.append(confidence)
                        else:
                            streetmaps_classification_list.append(np.NaN)
                            streetmaps_confidence_list.append(np.NaN)
                    except Exception as e:
                        logging.error(f"Error processing streetmaps image: {e}")
                        streetmaps_classification_list.append(np.NaN)
                        streetmaps_confidence_list.append(np.NaN)
                else:
                    streetmaps_classification_list.append(np.NaN)
                    streetmaps_confidence_list.append(np.NaN)

        # Add the prediction results to the DataFrame
        df_fincas_valoradas['catastro_classification'] = catastro_classification_list
        df_fincas_valoradas['catastro_confidence'] = catastro_confidence_list
        df_fincas_valoradas['streetmaps_classification'] = streetmaps_classification_list
        df_fincas_valoradas['streetmaps_confidence'] = streetmaps_confidence_list

        return df_fincas_valoradas
    
def final_step_classification(df_fincas_valoradas):
        #load data
    BUCKET_NAME = 'sophiq_static_files'

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Download the Parquet file from the bucket into memory
    blob = bucket.blob('all_fincas.parquet')
    parquet_data = BytesIO()
    blob.download_to_file(parquet_data)
    parquet_data.seek(0)  # Move to the beginning of the buffer
    # Read the data into a Pandas DataFrame
    all_fincas = pd.read_parquet(parquet_data, engine='pyarrow')

    df = df_fincas_valoradas.copy()

    df = df.merge(all_fincas,on='record_id',how='left')
    all_fincas.rename(columns={'Tipo Finca AI':'Tipo Finca AI old'},inplace=True)
    df['streetmaps_confidence'] = pd.to_numeric(df['streetmaps_confidence'], errors='coerce')*100
    df['zone'] = df['Barrio'].str[0]

    df['Tipo Finca AI'] = df['streetmaps_classification'].str.capitalize()

    df['catastro_confidence'] = pd.to_numeric(df['catastro_confidence'], errors='coerce')
    df['streetmaps_confidence'] = pd.to_numeric(df['streetmaps_confidence'], errors='coerce')

    # Initialize all as moderna
    df['Tipo Finca AI'] = 'Moderna'

    # Rule 1: Buildings before 1950 are clasica by default
    mask_old = (df['Año construcción'] <= 1950)
    df.loc[mask_old, 'Tipo Finca AI'] = 'Clasica'

    # Rule 2: Old buildings tagged as noclasica become clasica-moderna
    mask_old_noclasica = (mask_old & 
                        ((df.streetmaps_classification == 'noclasica') | 
                        (df.catastro_classification == 'noclasica')))
    df.loc[mask_old_noclasica, 'Tipo Finca AI'] = 'clasica-moderna'

    # Rule 3: Old buildings tagged as noclasica with high confidence become moderna
    mask_old_noclasica_confident = (mask_old & 
                                ((df.streetmaps_classification == 'noclasica') & 
                                    (df.streetmaps_confidence >= 80) |
                                    (df.catastro_classification == 'noclasica') & 
                                    (df.catastro_confidence >= 80)))
    df.loc[mask_old_noclasica_confident, 'Tipo Finca AI'] = 'Moderna'

    # Rule 4: New buildings tagged as clasica with high confidence
    mask_new_clasica = ((df['Año construcción'] > 1950) & 
                        ((df.streetmaps_classification == 'clasica') & 
                        (df.streetmaps_confidence >= 75) |
                        (df.catastro_classification == 'clasica') & 
                        (df.catastro_confidence >= 75)))
    df.loc[mask_new_clasica, 'Tipo Finca AI'] = 'Clasica'

    # Rule 5: Conflicting classifications with low confidence
    mask_conflicting = ((df.streetmaps_classification.notna()) & 
                    (df.catastro_classification.notna()) &
                    (df.streetmaps_classification != df.catastro_classification) &
                    (df.streetmaps_confidence < 70) &
                    (df.catastro_confidence < 70))
    df.loc[mask_conflicting, 'Tipo Finca AI'] = 'clasica-moderna'

    # Rule 4: New buildings tagged as clasica with high confidence
    mask_new_clasica = ((df['Año construcción'] > 1950) & 
                        ((df.streetmaps_classification == 'noclasica') & 
                        (df.streetmaps_confidence >= 75) |
                        (df.catastro_classification == 'noclasica') & 
                        (df.catastro_confidence >= 75)))
    df.loc[mask_new_clasica, 'Tipo Finca AI'] = 'Moderna'

    all_fincas = all_fincas.merge(df[['record_id','Tipo Finca AI']],on='record_id',how='left').drop(columns='Tipo Finca AI old')

    return all_fincas,df

from dotenv import load_dotenv; load_dotenv()

def main():
    print("Starting main function")

    # Read CSV File
    # data = pd.read_csv('updated_sampled_data.csv')

    # Initialize Class
    # Initialize GCS client
    storage_client = storage.Client()
    static_bucket = storage_client.bucket("sophiq_static_files")

    # # Use the base path since we moved the model
    gcp_model_path = "models/fine_tuned_classification_model.h5"  # Path inside the bucket
    local_model_path = "temp_model.h5"
    # Download model from GCP
    blob = static_bucket.blob(gcp_model_path)
    if not blob.exists():
        logging.error(f"Model not found at path: {gcp_model_path}")
        logging.info("Listing available model files:")
        for b in static_bucket.list_blobs():
            if '.h5' in b.name:
                logging.info(f"Found model: {b.name}")
        raise FileNotFoundError("Model file not found")

    blob.download_to_filename(local_model_path)
    logging.info(f"Successfully downloaded model from gs://sophiq_static_files/{gcp_model_path}")

    # Load the model using tensorflow
    logging.info("Successfully loaded model into memory")
    validator = GCPValidationClassification(local_model_path)

    # # Step 0: Prepare Dataset
    df_fincas_toSplit = validator.download_df_fincas_nameproperly()
    # # print(df_fincas_toSplit)

    #Step 1: Prepare GCP Dictionary
    folder_files, segmented_folder_files = validator.create_dic()
    # print(folder_files)
    #Step 2: Check for the missing Images
    print("[DEBUG] Starting image existence check for sampled data")
    missing_images_df = validator.check_images_for_sampled_data(folder_files, df_fincas_toSplit)
    #Step 3: Download Missing Images
    validator.download_missing_images(missing_images_df, df_fincas_toSplit)

    #Step 4: Model Retraining
    #Step 4.1: Appy Segmentation to directories
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Update this path to point to your YOLO model file
    yolo_model_path = "segment.pt"
    local_yolo_path = "segment_local.pt"
    yolo_blob = static_bucket.blob(yolo_model_path)

    try:
        yolo_blob.download_to_filename(local_yolo_path)
        logging.info(f"✅ Successfully downloaded YOLO model from GCP: gs://sophiq_static_files/{yolo_model_path}")
    except Exception as e:
        logging.error(f"❌ Error downloading YOLO model from GCP: {e}")
        return
    try:
        yolo_model = YOLO(local_yolo_path)
        logging.info("✅ YOLO model loaded successfully.")
        print("YOLO model loaded successfully.")
    except Exception as e:
        logging.error(f"❌ Failed to load YOLO model: {e}")
        return
    # List of input directories to process. Update these paths as needed.
    input_directories = [
        "gs://building_images_storage/fotos_fincas/clasica/",
        "gs://building_images_storage/fotos_fincas/noclasica/",
        "gs://building_images_storage/fotos_fincas/no_valorada/"
    ]

    # # Process the directories and save segmented images in new directories
    validator.segment_and_upload_directories(input_directories, yolo_model, segmented_folder_files, force_resegment=False)

    # # # Step 4.2: Apply Train Test Split
    validator.train_val_split_gcs()

    # Step 4.3: Apply Retraining
    train_dir = "training_fotos_fincas/train"  # e.g., "data/train"
    val_dir = "training_fotos_fincas/validation"  # e.g., "data/val"

    result = validator.retrain(train_dir, val_dir)

    if result is None:
        logging.info("Training skipped - will try again later.")
        model_path = local_model_path  # Use existing local model
    else:
        model_path, history = result
        logging.info("Training completed successfully")
        
        # Download the newly trained model if it's a GCS path
        if model_path.startswith("gs://"):
            bucket_name = model_path.split("/")[2]
            blob_path = "/".join(model_path.split("/")[3:])
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.download_to_filename(local_model_path)
            logging.info(f"Downloaded newly trained model from {model_path} to {local_model_path}")
        
    model = tf.keras.models.load_model(local_model_path)

    dict_fotos_fincas = validator.run_dict_fotos_fincas(df_fincas_toSplit)
    df_fincas_paths = validator.generate_df_streetmaps_catastro_paths(dict_fotos_fincas)

    img_height = 224
    img_width = 224
    class_names = ['clasica', 'noclasica']
    df_fincas_valoradas = validator.apply_model_generate_df_classification_confidence(df_fincas_paths, model,
                                                                                      img_height, img_width,
                                                                                      class_names)

    all_fincas,df_fincas_valoradas = final_step_classification(df_fincas_valoradas)

    bucket_name = 'sophiq_static_files'

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Assuming `df_quadrants` is already defined
    # Replace any NaN values with None and ensure the DataFrame is ready for export
    df = all_fincas.where(pd.notnull(all_fincas), None).replace(np.nan, None)

    # Convert the DataFrame to a Parquet file in memory
    parquet_data = BytesIO()
    df.to_parquet(parquet_data, index=False, engine='pyarrow')
    parquet_data.seek(0)  # Reset buffer to the beginning for reading

    # Upload the Parquet file to the specified GCS bucket
    blob = bucket.blob('all_fincas.parquet')
    blob.upload_from_file(parquet_data, content_type='application/octet-stream')

    print(f"DataFrame is successfully uploaded to all_fincas.parquet in the {bucket_name} bucket.")

    # Assuming `df_quadrants` is already defined
    # Replace any NaN values with None and ensure the DataFrame is ready for export
    df = df_fincas_valoradas.where(pd.notnull(df_fincas_valoradas), None).replace(np.nan, None)

    # Convert the DataFrame to a Parquet file in memory
    parquet_data = BytesIO()
    df.to_parquet(parquet_data, index=False, engine='pyarrow')
    parquet_data.seek(0)  # Reset buffer to the beginning for reading

    # Upload the Parquet file to the specified GCS bucket
    blob = bucket.blob('df_fincas_valoradas.parquet')
    blob.upload_from_file(parquet_data, content_type='application/octet-stream')

    print(f"DataFrame is successfully uploaded to df_fincas_valoradas.parquet in the {bucket_name} bucket.")

    return 'Success!'



if __name__ == '__main__':
    print(main())