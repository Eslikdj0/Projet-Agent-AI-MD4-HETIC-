"""
Data Cleaning & Processing Script
Purpose: Clean and process tmdb_5000_credits.csv data
- Parses JSON cast/crew data
- Handles missing values
- Creates both raw and clean versions
- Normalizes data types and formats
"""

import pandas as pd
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_FILE = "data/tmdb_5000_credits.csv"
OUTPUT_DIR = "data/output"
RAW_OUTPUT = "data/output/tmdb_credits_raw.csv"
CLEAN_OUTPUT = "data/output/tmdb_credits_clean.csv"
CLEAN_JSON_OUTPUT = "data/output/tmdb_credits_clean.json"

class DataCleaner:
    """Main data cleaning class"""
    
    def __init__(self, input_file):
        self.input_file = input_file
        self.raw_data = None
        self.clean_data = None
        self.cleaning_stats = {
            "original_records": 0,
            "records_after_validation": 0,
            "records_dropped": 0,
            "invalid_json_records": 0
        }
    
    def create_output_dir(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.info(f"Output directory ready: {OUTPUT_DIR}")
    
    def load_data(self):
        """Load CSV file"""
        try:
            self.raw_data = pd.read_csv(self.input_file, dtype={'movie_id': 'int64', 'title': 'str'})
            self.cleaning_stats["original_records"] = len(self.raw_data)
            logger.info(f"✓ Loaded {len(self.raw_data)} records from {self.input_file}")
            return True
        except Exception as e:
            logger.error(f"✗ Error loading file: {e}")
            return False
    
    def parse_json_field(self, value, default=None):
        """Safely parse JSON field"""
        if pd.isna(value) or value == '':
            return default if default is not None else []
        try:
            if isinstance(value, str):
                return json.loads(value)
            return value
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON detected: {value[:50]}...")
            return default if default is not None else []
    
    def validate_records(self):
        """
        Validate records and remove invalid entries
        Requirements: title and movie_id must exist
        """
        logger.info("\n" + "="*60)
        logger.info("VALIDATION PHASE")
        logger.info("="*60)
        
        # Create a copy for processing
        df = self.raw_data.copy()
        
        # Check for missing titles or movie IDs
        initial_count = len(df)
        
        # Remove rows without title
        df = df[df['title'].notna() & (df['title'].str.len() > 0)]
        dropped_no_title = initial_count - len(df)
        
        # Remove rows without movie_id
        df = df[df['movie_id'].notna()]
        dropped_no_id = len(df) - (initial_count - dropped_no_title)
        
        self.cleaning_stats["records_after_validation"] = len(df)
        self.cleaning_stats["records_dropped"] = initial_count - len(df)
        
        logger.info(f"Initial records: {initial_count}")
        logger.info(f"Records dropped (no title): {dropped_no_title}")
        logger.info(f"Records dropped (no ID): {dropped_no_id}")
        logger.info(f"Valid records: {len(df)}")
        
        return df
    
    def parse_cast_crew(self, df):
        """Parse cast and crew JSON data"""
        logger.info("\n" + "="*60)
        logger.info("PARSING CAST & CREW")
        logger.info("="*60)
        
        # Parse cast data
        df['cast_parsed'] = df['cast'].apply(self.parse_json_field)
        df['crew_parsed'] = df['crew'].apply(self.parse_json_field)
        
        # Extract cast member names
        def extract_cast_names(cast_list):
            try:
                if isinstance(cast_list, list):
                    return [actor.get('name', '') for actor in cast_list if 'name' in actor]
                return []
            except:
                return []
        
        def extract_crew_names(crew_list):
            try:
                if isinstance(crew_list, list):
                    return [crew.get('name', '') for crew in crew_list if 'name' in crew]
                return []
            except:
                return []
        
        df['cast_names'] = df['cast_parsed'].apply(extract_cast_names)
        df['crew_names'] = df['crew_parsed'].apply(extract_crew_names)
        
        # Count of cast/crew members
        df['cast_count'] = df['cast_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['crew_count'] = df['crew_parsed'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        logger.info("✓ Cast and crew parsed successfully")
        logger.info(f"  Average cast members: {df['cast_count'].mean():.2f}")
        logger.info(f"  Average crew members: {df['crew_count'].mean():.2f}")
        
        return df
    
    def normalize_data(self, df):
        """
        Normalize data for the clean version
        - Lowercase title for consistency (optional)
        - Standardize data types
        """
        logger.info("\n" + "="*60)
        logger.info("DATA NORMALIZATION")
        logger.info("="*60)
        
        df_clean = df.copy()
        
        # Normalize title (trim whitespace)
        df_clean['title'] = df_clean['title'].str.strip()
        
        # Ensure movie_id is integer
        df_clean['movie_id'] = df_clean['movie_id'].astype('int64')
        
        # Convert cast/crew names to JSON strings for storage
        df_clean['cast_names_json'] = df_clean['cast_names'].apply(json.dumps)
        df_clean['crew_names_json'] = df_clean['crew_names'].apply(json.dumps)
        
        logger.info("✓ Data normalized successfully")
        
        return df_clean
    
    def create_clean_version(self):
        """Create cleaned version of data"""
        logger.info("\n" + "="*60)
        logger.info("CREATING CLEAN VERSION")
        logger.info("="*60)
        
        # Validate records
        validated_df = self.validate_records()
        
        # Parse cast/crew
        parsed_df = self.parse_cast_crew(validated_df)
        
        # Normalize
        clean_df = self.normalize_data(parsed_df)
        
        # Select columns for clean output
        self.clean_data = clean_df[[
            'movie_id',
            'title',
            'cast_count',
            'crew_count',
            'cast_names_json',
            'crew_names_json'
        ]].copy()
        
        # Rename columns for clarity
        self.clean_data.columns = [
            'movie_id',
            'title',
            'cast_members_count',
            'crew_members_count',
            'cast_names',
            'crew_names'
        ]
        
        logger.info(f"✓ Clean dataset created with {len(self.clean_data)} records")
        logger.info(f"  Shape: {self.clean_data.shape}")
        
        return self.clean_data
    
    def save_outputs(self):
        """Save both raw and clean versions"""
        logger.info("\n" + "="*60)
        logger.info("SAVING DATA")
        logger.info("="*60)
        
        try:
            # Save raw data
            self.raw_data.to_csv(RAW_OUTPUT, index=False, encoding='utf-8')
            logger.info(f"✓ Raw data saved to: {RAW_OUTPUT}")
            
            # Save clean data as CSV
            self.clean_data.to_csv(CLEAN_OUTPUT, index=False, encoding='utf-8')
            logger.info(f"✓ Clean data saved to: {CLEAN_OUTPUT}")
            
            # Save clean data as JSON (more readable for nested data)
            clean_json = self.clean_data.to_dict('records')
            with open(CLEAN_JSON_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(clean_json, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Clean data (JSON) saved to: {CLEAN_JSON_OUTPUT}")
            
            return True
        except Exception as e:
            logger.error(f"✗ Error saving data: {e}")
            return False
    
    def print_statistics(self):
        """Print cleaning statistics"""
        logger.info("\n" + "="*60)
        logger.info("CLEANING STATISTICS")
        logger.info("="*60)
        
        for key, value in self.cleaning_stats.items():
            logger.info(f"{key}: {value}")
        
        if self.cleaning_stats["original_records"] > 0:
            retention_rate = (self.cleaning_stats["records_after_validation"] / self.cleaning_stats["original_records"]) * 100
            logger.info(f"Data retention rate: {retention_rate:.2f}%")
    
    def run(self):
        """Execute full cleaning pipeline"""
        logger.info("\n" + "█"*60)
        logger.info("TMDB 5000 CREDITS DATA CLEANING PIPELINE")
        logger.info("█"*60)
        
        self.create_output_dir()
        
        if not self.load_data():
            return False
        
        self.create_clean_version()
        
        if not self.save_outputs():
            return False
        
        self.print_statistics()
        
        logger.info("\n" + "█"*60)
        logger.info("CLEANING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("█"*60)
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return True

def main():
    """Main execution"""
    cleaner = DataCleaner(INPUT_FILE)
    cleaner.run()

if __name__ == "__main__":
    main()
