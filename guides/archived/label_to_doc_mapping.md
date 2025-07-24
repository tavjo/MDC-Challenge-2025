# Label to Document Mapping Guide

This guide provides step-by-step instructions for creating a class that loads labels, conducts basic data quality checks, and maps documents to their full-text file paths.

## **1. Dependencies for Label Mapping**

**⚠️ IMPORTANT**: This setup focuses on **essential dependencies** for label loading and document path mapping operations.

```bash
# activate virtual environment
source .venv/bin/activate

# core data manipulation
uv add pandas numpy

# visualization for class balance charts
uv add matplotlib

# file path operations (pathlib is built-in, no installation needed)
# os and glob are also built-in modules
```

---

## **2. Implementation Structure**

The implementation consists of:
- **Class location**: `src/label_mapper.py` 
- **Usage location**: Jupyter notebooks in `notebooks/` directory
- **Core functionality**: Label loading, document path mapping, data quality checks

---

## **3. Step-by-Step Implementation**


The complete `LabelMapper` class has been implemented in `src/label_mapper.py` with all functionality as specified below.

### **Step 1: Create the LabelMapper Class Structure** ✅ **COMPLETED**

Create `src/label_mapper.py` with the following class structure:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
import warnings

class LabelMapper:
    """
    Handles loading labels, mapping documents to file paths, and conducting basic QC checks.
    
    Expected class distribution:
    - Secondary: ~44%
    - Missing: ~30% 
    - Primary: ~26%
    """
    
    def __init__(self, data_dir: str = "Data"):
        """
        Initialize the LabelMapper.
        
        Args:
            data_dir (str): Path to the data directory containing labels and documents
        """
        self.data_dir = Path(data_dir)
        self.labels_df = None
        self.document_inventory = None
        self.missing_files = []
        
    # Methods will be implemented in subsequent steps
```

### **Step 2: Implement Label Loading Method** ✅ **COMPLETED**

Add the label loading method:

```python
def load_labels(self, labels_file: str = "train_labels.csv") -> pd.DataFrame:
    """
    Load labels from CSV file.
    
    Args:
        labels_file (str): Name of the labels CSV file
        
    Returns:
        pd.DataFrame: Loaded labels with columns: article_id, dataset_id, type
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
        ValueError: If required columns are missing
    """
    labels_path = self.data_dir / labels_file
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Load labels
    self.labels_df = pd.read_csv(labels_path)
    
    # Validate required columns
    required_cols = ['article_id', 'dataset_id', 'type']
    missing_cols = [col for col in required_cols if col not in self.labels_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Loaded {len(self.labels_df)} labels from {labels_file}")
    print(f"📊 Columns: {list(self.labels_df.columns)}")
    
    return self.labels_df
```

### **Step 3: Implement Document Path Inventory Method** ✅ **COMPLETED**

Add the method to inventory full-text file paths:

```python
def inventory_document_paths(self, 
                           pdf_dir: str = "pdfs", 
                           xml_dir: str = "xmls") -> pd.DataFrame:
    """
    Create inventory of available PDF and XML files and map to article IDs.
    
    Args:
        pdf_dir (str): Directory name containing PDF files
        xml_dir (str): Directory name containing XML files  
        
    Returns:
        pd.DataFrame: Document inventory with paths and availability flags
    """
    if self.labels_df is None:
        raise ValueError("Labels must be loaded first. Call load_labels() method.")
    
    # Initialize inventory
    inventory_data = []
    
    # Set up directory paths
    pdf_path = self.data_dir / pdf_dir
    xml_path = self.data_dir / xml_dir
    
    print(f"🔍 Scanning directories:")
    print(f"  📁 PDF: {pdf_path}")
    print(f"  📁 XML: {xml_path}")
    
    # Create file lookup dictionaries for faster searching
    pdf_files = {}
    xml_files = {}
    
    if pdf_path.exists():
        for pdf_file in pdf_path.glob("*.pdf"):
            # Extract article ID from filename (assuming format like "article_123.pdf")
            article_id = pdf_file.stem
            pdf_files[article_id] = pdf_file
    
    if xml_path.exists():
        for xml_file in xml_path.glob("*.xml"):
            # Extract article ID from filename (assuming format like "article_123.xml")
            article_id = xml_file.stem
            xml_files[article_id] = xml_file
    
    # Process each article in labels
    self.missing_files = []
    
    for _, row in self.labels_df.iterrows():
        article_id = str(row['article_id'])
        
        # Check for PDF
        pdf_available = article_id in pdf_files
        pdf_path_str = str(pdf_files[article_id]) if pdf_available else None
        
        # Check for XML
        xml_available = article_id in xml_files
        xml_path_str = str(xml_files[article_id]) if xml_available else None
        
        # Track missing files
        if not (pdf_available or xml_available):
            self.missing_files.append(article_id)
        
        inventory_data.append({
            'article_id': article_id,
            'dataset_id': row['dataset_id'],
            'type': row['type'],
            'pdf_available': pdf_available,
            'pdf_path': pdf_path_str,
            'xml_available': xml_available,
            'xml_path': xml_path_str,
            'has_fulltext': pdf_available or xml_available
        })
    
    self.document_inventory = pd.DataFrame(inventory_data)
    
    # Summary statistics
    total_articles = len(self.document_inventory)
    pdf_count = self.document_inventory['pdf_available'].sum()
    xml_count = self.document_inventory['xml_available'].sum()
    fulltext_count = self.document_inventory['has_fulltext'].sum()
    missing_count = len(self.missing_files)
    
    print(f"\n📋 Document Inventory Summary:")
    print(f"  Total articles: {total_articles}")
    print(f"  📄 PDFs available: {pdf_count} ({pdf_count/total_articles*100:.1f}%)")
    print(f"  🔖 XMLs available: {xml_count} ({xml_count/total_articles*100:.1f}%)")
    print(f"  📚 Has full-text: {fulltext_count} ({fulltext_count/total_articles*100:.1f}%)")
    print(f"  ❌ Missing files: {missing_count} ({missing_count/total_articles*100:.1f}%)")
    
    if self.missing_files:
        print(f"\n⚠️  First 10 missing article IDs: {self.missing_files[:10]}")
    
    return self.document_inventory
```

### **Step 4: Implement Basic Quality Checks Method** ✅ **COMPLETED**

Add the method for conducting basic data quality checks:

```python
def conduct_basic_checks(self, show_plots: bool = True) -> Dict:
    """
    Conduct basic data quality checks on labels and document availability.
    
    Args:
        show_plots (bool): Whether to display visualization plots
        
    Returns:
        Dict: Summary of quality check results
    """
    if self.labels_df is None:
        raise ValueError("Labels must be loaded first. Call load_labels() method.")
    
    results = {}
    
    # 1. Check for duplicates
    duplicate_articles = self.labels_df['article_id'].duplicated().sum()
    results['duplicate_articles'] = duplicate_articles
    
    if duplicate_articles > 0:
        print(f"⚠️  Found {duplicate_articles} duplicate article IDs")
        duplicate_ids = self.labels_df[self.labels_df['article_id'].duplicated(keep=False)]
        print(f"   Sample duplicates: {duplicate_ids['article_id'].head().tolist()}")
    else:
        print("✅ No duplicate article IDs found")
    
    # 2. Check for null values
    null_counts = self.labels_df.isnull().sum()
    results['null_counts'] = null_counts.to_dict()
    
    print(f"\n🔍 Null value check:")
    for col, null_count in null_counts.items():
        if null_count > 0:
            print(f"   ⚠️  {col}: {null_count} nulls ({null_count/len(self.labels_df)*100:.1f}%)")
        else:
            print(f"   ✅ {col}: No nulls")
    
    # 3. Class balance analysis
    class_counts = self.labels_df['type'].value_counts()
    class_percentages = self.labels_df['type'].value_counts(normalize=True) * 100
    results['class_distribution'] = class_counts.to_dict()
    results['class_percentages'] = class_percentages.to_dict()
    
    print(f"\n📊 Class Distribution:")
    for class_type, count in class_counts.items():
        percentage = class_percentages[class_type]
        print(f"   {class_type}: {count} ({percentage:.1f}%)")
    
    # Expected distribution check
    expected_dist = {'Secondary': 44, 'Missing': 30, 'Primary': 26}
    print(f"\n🎯 Expected vs Actual Distribution:")
    for class_type, expected_pct in expected_dist.items():
        if class_type in class_percentages:
            actual_pct = class_percentages[class_type]
            diff = abs(actual_pct - expected_pct)
            status = "✅" if diff <= 5 else "⚠️" if diff <= 10 else "❌"
            print(f"   {status} {class_type}: Expected {expected_pct}%, Actual {actual_pct:.1f}% (Δ{diff:.1f}%)")
        else:
            print(f"   ❌ {class_type}: Expected {expected_pct}%, Not found in data")
    
    # 4. Visualization
    if show_plots:
        self._create_class_balance_chart(class_counts)
    
    return results
```

### **Step 5: Implement Visualization Helper Method** ✅ **COMPLETED**

Add the private method for creating visualizations:

```python
def _create_class_balance_chart(self, class_counts: pd.Series):
    """Create bar chart showing class balance."""
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(class_counts.index, class_counts.values, 
                   color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.8)
    
    # Add percentage labels on bars
    total = class_counts.sum()
    for bar, count in zip(bars, class_counts.values):
        height = bar.get_height()
        percentage = (count / total) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + total*0.01,
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('Label Type Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Label Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add expected distribution as reference
    expected_counts = {
        'Primary': total * 0.26,
        'Secondary': total * 0.44, 
        'Missing': total * 0.30
    }
    
    x_positions = range(len(class_counts))
    expected_values = [expected_counts.get(label, 0) for label in class_counts.index]
    plt.plot(x_positions, expected_values, 'ro--', alpha=0.7, 
             linewidth=2, markersize=8, label='Expected Distribution')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
```

### **Step 6: Add Utility Methods** ✅ **COMPLETED**

Add helpful utility methods:

```python
def get_summary_stats(self) -> Dict:
    """Get comprehensive summary statistics."""
    if self.labels_df is None or self.document_inventory is None:
        raise ValueError("Must load labels and inventory documents first.")
    
    return {
        'total_articles': len(self.labels_df),
        'unique_datasets': self.labels_df['dataset_id'].nunique(),
        'class_distribution': self.labels_df['type'].value_counts().to_dict(),
        'documents_with_fulltext': self.document_inventory['has_fulltext'].sum(),
        'missing_documents': len(self.missing_files),
        'pdf_availability': self.document_inventory['pdf_available'].sum(),
        'xml_availability': self.document_inventory['xml_available'].sum()
    }

def export_missing_files(self, output_file: str = "missing_files.txt"):
    """Export list of missing files for QC purposes."""
    if not self.missing_files:
        print("✅ No missing files to export")
        return
    
    output_path = self.data_dir / output_file
    with open(output_path, 'w') as f:
        f.write("# Missing Article Files\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n")
        f.write(f"# Total missing: {len(self.missing_files)}\n\n")
        for article_id in self.missing_files:
            f.write(f"{article_id}\n")
    
    print(f"📝 Exported {len(self.missing_files)} missing article IDs to {output_path}")

def get_document_path(self, article_id: str, prefer_pdf: bool = False) -> Optional[str]:
    """
    Get the file path for a specific article.
    
    Args:
        article_id (str): The article ID to look up
        prefer_pdf (bool): Whether to prefer PDF over XML when both available
        
    Returns:
        Optional[str]: File path if available, None otherwise
    """
    if self.document_inventory is None:
        raise ValueError("Must inventory documents first.")
    
    row = self.document_inventory[self.document_inventory['article_id'] == str(article_id)]
    
    if row.empty:
        return None
    
    row = row.iloc[0]
    
    if prefer_pdf and row['pdf_available']:
        return row['pdf_path']
    elif row['xml_available']:
        return row['xml_path']
    elif row['pdf_available']:
        return row['pdf_path']
    else:
        return None
```

---

## **4. Usage in Notebooks**

### **Basic Usage Example**

```python
# In your Jupyter notebook (notebooks/label_mapping_analysis.ipynb)

# Add src to path
import sys
sys.path.append('../src')

from label_mapper import LabelMapper

# Initialize mapper
mapper = LabelMapper(data_dir="../Data")

# Load labels
labels_df = mapper.load_labels("train_labels.csv")

# Inventory document paths (uses train/PDF and train/XML by default)
inventory_df = mapper.inventory_document_paths()

# Conduct quality checks
qc_results = mapper.conduct_basic_checks(show_plots=True)

# Get summary statistics
summary = mapper.get_summary_stats()
print(summary)

# Export missing files for QC
mapper.export_missing_files("missing_articles_qc.txt")
```

### **Advanced Usage**

```python
# Access specific document paths
pdf_path = mapper.get_document_path("article_123", prefer_pdf = False)

# Filter inventory for specific conditions
articles_with_pdf = mapper.document_inventory[
    mapper.document_inventory['pdf_available'] == True
]

# Analyze by dataset
dataset_stats = mapper.document_inventory.groupby('dataset_id').agg({
    'has_fulltext': ['count', 'sum'],
    'pdf_available': 'sum',
    'xml_available': 'sum'
})
```

---

## **5. Expected Outputs**

When running the complete pipeline, expect:

- ✅ **Labels loaded** with validation of required columns (1,028 labels loaded)
- 📊 **Class distribution** approximately: 44% Secondary, 30% Missing, 26% Primary  
- 📋 **Document inventory** showing PDF/XML availability:
  - 📄 **100% PDF availability** (1,028/1,028 files found)
  - 🔖 **85.1% XML availability** (875/1,028 files found)
  - 📚 **100% full-text coverage** (all articles have at least PDF available)
  - ❌ **0 missing files** for this dataset
- 📈 **Visualization** of class balance with expected vs actual comparison

**✅ IMPLEMENTATION VERIFIED**: The LabelMapper successfully matches article IDs directly to filenames using exact DOI-based naming (e.g., `10.1002_2017jc013030.pdf`).

The implementation provides a robust foundation for document-label mapping with comprehensive error handling and quality checks. 