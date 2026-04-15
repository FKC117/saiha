"""
Utility functions for working with datasets in the ChatFlow project.
Ported and refactored from Quantly.
"""
import pandas as pd
import os
import uuid
import logging
from django.conf import settings
from ..models import Dataset, DatasetColumn
from .storage_manager_parquet import DatasetStorageManager
from .dataset_processor import DatasetProcessor

logger = logging.getLogger(__name__)

def load_dataset_data(dataset_id, columns=None, user=None):
    """
    Load dataset data using Parquet with column projection.
    Pass `user` to enforce ownership: only that user's dataset is returned.
    """
    try:
        qs = Dataset.objects.filter(id=dataset_id)
        if user is not None:
            qs = qs.filter(user=user)
        dataset = qs.get()
        storage_manager = DatasetStorageManager()
        return storage_manager.load_processed_file(dataset.user.id, dataset.id, columns=columns)
    except Dataset.DoesNotExist:
        raise ValueError(f"Dataset {dataset_id} not found or access denied.")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def save_dataframe_as_dataset(df: pd.DataFrame, original_dataset: Dataset, suffix: str) -> Dataset:
    """
    Saves a modified DataFrame as a new versioned dataset (Lineage tracking).
    """
    new_id = uuid.uuid4()
    # Consistent naming: "Original Name (Suffix)"
    new_name = f"{original_dataset.name} ({suffix})"
    
    storage_manager = DatasetStorageManager()
    processor = DatasetProcessor()
    
    # 1. Save physical file (Parquet)
    file_path = storage_manager.save_processed_file(df, original_dataset.user.id, new_id, 'parquet')
    
    # 2. Extract column metadata
    columns_metadata = processor.get_column_metadata(df)
    
    # 3. Save JSON metadata/preview
    metadata = {
        'dataset_id': str(new_id),
        'name': new_name,
        'parent_dataset_id': str(original_dataset.id),
        'created_via': f"Transformation: {suffix}"
    }
    metadata_path = storage_manager.save_metadata(metadata, original_dataset.user.id, new_id, df=df)
    
    # 4. Create database records
    new_dataset = Dataset.objects.create(
        id=new_id,
        user=original_dataset.user,
        name=new_name,
        original_filename=original_dataset.original_filename,
        file_type=original_dataset.file_type,
        storage_format='parquet',
        file_size=os.path.getsize(file_path),
        rows_count=len(df),
        columns_count=len(df.columns),
        is_processed=True,
        processed_file_path=storage_manager.get_relative_path(file_path),
        metadata_file_path=storage_manager.get_relative_path(metadata_path),
        parent_dataset=original_dataset
    )
    
    # 5. Create DatasetColumn records
    for col_meta in columns_metadata:
        DatasetColumn.objects.create(
            dataset=new_dataset,
            column_name=col_meta['column_name'],
            column_index=col_meta['column_index'],
            data_type=col_meta['data_type'],
            null_count=col_meta['null_count'],
            unique_count=col_meta['unique_count'],
            sample_values=col_meta.get('sample_values', [])
        )
        
    return new_dataset
