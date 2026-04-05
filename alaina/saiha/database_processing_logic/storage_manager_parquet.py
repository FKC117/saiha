import os
import json
import uuid
from django.conf import settings
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DatasetStorageManager:
    """
    Handles file storage for datasets with Parquet format for maximum performance.
    Ported from Quantly to ChatFlow.
    """
    
    def __init__(self):
        self.base_path = os.path.join(settings.MEDIA_ROOT, 'datasets')
        self.ensure_base_directory()
    
    def ensure_base_directory(self):
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)
    
    def get_user_directory(self, user_id):
        user_dir = os.path.join(self.base_path, str(user_id))
        if not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)
        return user_dir
    
    def get_dataset_directory(self, user_id, dataset_id):
        user_dir = self.get_user_directory(user_id)
        dataset_dir = os.path.join(user_dir, str(dataset_id))
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir
    
    def save_processed_file(self, df, user_id, dataset_id, file_type):
        dataset_dir = self.get_dataset_directory(user_id, dataset_id)
        filename = f"processed_{dataset_id}.parquet"
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            # Save as Parquet (fastest and most efficient format)
            tmp_filename = f"processed_{dataset_id}.parquet.tmp"
            tmp_path = os.path.join(dataset_dir, tmp_filename)
            df.to_parquet(
                tmp_path,
                index=False,
                engine='pyarrow',
                compression='snappy'
            )
            os.replace(tmp_path, file_path)
            logger.info(f"Saved dataset as Parquet: {file_path}")
            return file_path

        except Exception as e:
            logger.error(f"Error saving Parquet file: {e}, falling back to CSV")
            filename = f"processed_{dataset_id}.csv"
            file_path = os.path.join(dataset_dir, filename)
            df.to_csv(file_path, index=False, encoding='utf-8')
            return file_path
    
    def load_processed_file(self, user_id, dataset_id, columns=None):
        dataset_dir = self.get_dataset_directory(user_id, dataset_id)
        parquet_file = os.path.join(dataset_dir, f"processed_{dataset_id}.parquet")
        
        if os.path.exists(parquet_file):
            try:
                return pd.read_parquet(parquet_file, engine='pyarrow', columns=columns)
            except Exception as e:
                logger.warning(f"Error loading Parquet file: {e}, trying CSV fallback")
        
        csv_file = os.path.join(dataset_dir, f"processed_{dataset_id}.csv")
        if os.path.exists(csv_file):
            return pd.read_csv(csv_file)
        
        raise FileNotFoundError(f"No processed file found for dataset {dataset_id}")
    
    def save_metadata(self, metadata, user_id, dataset_id, df=None, preview_rows=10):
        dataset_dir = self.get_dataset_directory(user_id, dataset_id)
        filename = f"dataset_info_{dataset_id}.json"
        file_path = os.path.join(dataset_dir, filename)
        
        preview_data = None
        if df is not None:
            preview_df = df.head(preview_rows)
            preview_data = {
                'columns': list(preview_df.columns),
                'data': preview_df.to_dict('records'),
                'total_rows': len(df),
                'preview_rows': len(preview_df)
            }
        
        metadata.update({
            'dataset_id': str(dataset_id),
            'created_at': pd.Timestamp.now().isoformat(),
            'storage_path': dataset_dir,
            'storage_format': 'parquet',
            'preview_data': preview_data
        })
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return file_path
    
    def load_metadata(self, user_id, dataset_id):
        dataset_dir = self.get_dataset_directory(user_id, dataset_id)
        filename = f"dataset_info_{dataset_id}.json"
        file_path = os.path.join(dataset_dir, filename)
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def delete_dataset_files(self, user_id, dataset_id):
        dataset_dir = self.get_dataset_directory(user_id, dataset_id)
        if os.path.exists(dataset_dir):
            import shutil
            shutil.rmtree(dataset_dir)
            return True
        return False
    
    def get_relative_path(self, absolute_path):
        if absolute_path.startswith(settings.MEDIA_ROOT):
            return os.path.relpath(absolute_path, settings.MEDIA_ROOT)
        return absolute_path
