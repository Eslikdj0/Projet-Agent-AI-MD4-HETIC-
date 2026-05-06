"""
Data Inspection & Quality Check Script
Purpose: Analyze the tmdb_5000_credits.csv file structure and identify data quality issues
"""

import pandas as pd
import json
import os
from datetime import datetime

# Configuration
DATA_FILE = "data/tmdb_5000_credits.csv"
REPORT_DIR = "data/reports"

def create_report_dir():
    """Create reports directory if it doesn't exist"""
    os.makedirs(REPORT_DIR, exist_ok=True)

def check_basic_info(df):
    """Check basic file information"""
    print("\n" + "="*60)
    print("BASIC FILE INFORMATION")
    print("="*60)
    print(f"File: {DATA_FILE}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    print(f"\nData Types:\n{df.dtypes}")
    
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist()
    }

def check_missing_values(df):
    """Check for missing or null values"""
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    report = pd.DataFrame({
        "Column": missing.index,
        "Missing Count": missing.values,
        "Missing %": missing_pct.values
    })
    
    print(report.to_string(index=False))
    
    return report

def check_empty_strings(df):
    """Check for empty string values"""
    print("\n" + "="*60)
    print("EMPTY STRINGS ANALYSIS")
    print("="*60)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_count = (df[col].str.len() == 0).sum()
            empty_pct = (empty_count / len(df)) * 100
            if empty_count > 0:
                print(f"{col}: {empty_count} empty strings ({empty_pct:.2f}%)")
    
def check_json_validity(df, col):
    """Check if JSON columns can be parsed"""
    print(f"\nChecking JSON validity for column: {col}")
    invalid_count = 0
    
    for idx, value in df[col].items():
        try:
            if isinstance(value, str) and value.strip():
                json.loads(value)
        except json.JSONDecodeError:
            invalid_count += 1
    
    print(f"  Invalid JSON entries: {invalid_count}/{len(df)}")
    return invalid_count

def check_cast_crew_content(df):
    """Analyze cast and crew content"""
    print("\n" + "="*60)
    print("CAST & CREW CONTENT ANALYSIS")
    print("="*60)
    
    for col in ['cast', 'crew']:
        print(f"\n{col.upper()}:")
        
        # Check JSON validity
        invalid = check_json_validity(df, col)
        
        # Sample analysis
        sample_data = None
        for value in df[col]:
            if isinstance(value, str) and value.strip():
                try:
                    sample_data = json.loads(value)
                    if sample_data:
                        break
                except:
                    pass
        
        if sample_data:
            print(f"  Sample entry (first item): {json.dumps(sample_data[:1] if isinstance(sample_data, list) else sample_data, indent=2)[:200]}...")
            if isinstance(sample_data, list):
                print(f"  Average items per row: {pd.Series([len(json.loads(x)) if isinstance(x, str) and x.strip() else 0 for x in df[col]]).mean():.2f}")

def check_title_validity(df):
    """Check title column validity"""
    print("\n" + "="*60)
    print("TITLE VALIDITY CHECK")
    print("="*60)
    
    empty_titles = (df['title'].str.len() == 0).sum()
    null_titles = df['title'].isnull().sum()
    
    print(f"Empty titles: {empty_titles}")
    print(f"Null titles: {null_titles}")
    print(f"Valid titles: {len(df) - empty_titles - null_titles}")
    print(f"Total records with valid title: {len(df[df['title'].notna() & (df['title'].str.len() > 0)])}")

def check_movie_id(df):
    """Check movie_id uniqueness and validity"""
    print("\n" + "="*60)
    print("MOVIE ID VALIDATION")
    print("="*60)
    
    print(f"Unique movie IDs: {df['movie_id'].nunique()}")
    print(f"Total records: {len(df)}")
    print(f"Duplicate IDs: {len(df) - df['movie_id'].nunique()}")
    print(f"ID range: {df['movie_id'].min()} to {df['movie_id'].max()}")

def generate_quality_score(df):
    """Calculate overall data quality score"""
    print("\n" + "="*60)
    print("DATA QUALITY SCORE")
    print("="*60)
    
    total_cells = df.shape[0] * df.shape[1]
    null_cells = df.isnull().sum().sum()
    empty_string_cells = (df.map(lambda x: isinstance(x, str) and len(x) == 0)).sum().sum()
    
    quality_cells = total_cells - null_cells - empty_string_cells
    quality_score = (quality_cells / total_cells) * 100
    
    print(f"Total cells: {total_cells}")
    print(f"Valid cells: {quality_cells}")
    print(f"Invalid cells (null or empty): {null_cells + empty_string_cells}")
    print(f"Quality Score: {quality_score:.2f}%")
    
    return quality_score

def main():
    """Main execution"""
    print("\n" + "█"*60)
    print("TMDB 5000 CREDITS DATA CHECK")
    print("█"*60)
    
    create_report_dir()
    
    # Load data
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"\n✓ Successfully loaded {DATA_FILE}")
    except FileNotFoundError:
        print(f"\n✗ Error: File '{DATA_FILE}' not found")
        return
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        return
    
    # Run all checks
    check_basic_info(df)
    check_missing_values(df)
    check_empty_strings(df)
    check_cast_crew_content(df)
    check_title_validity(df)
    check_movie_id(df)
    quality_score = generate_quality_score(df)
    
    print("\n" + "="*60)
    print("CHECK COMPLETE")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
