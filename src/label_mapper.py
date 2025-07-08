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
    
    Key Features:
    - Distinguishes between label entries (rows) and unique articles
    - Categorizes articles for PDF->XML conversion workflow
    - Provides accurate file availability statistics
    
    Expected class distribution:
    - Secondary: ~44%
    - Missing: ~30% 
    - Primary: ~26%
    """
    TEMP_SUFFIX = '.part'
    
    def __init__(self, data_dir: str = "Data"):
        """
        Initialize the LabelMapper.
        
        Args:
            data_dir (str): Path to the data directory containing labels and documents
        """
        self.data_dir = Path(data_dir)
        self.labels_df = None
        self.document_inventory = None
        self.unique_articles_df = None
        self.file_availability_summary = None
        # self.output_dir = os.path.join(data_dir, self.TEMP_SUFFIX)
        
    def load_labels(self, labels_file: str = "train_labels.csv") -> pd.DataFrame:
        """
        Load labels from CSV file and compute unique article statistics.
        
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
        
        # Compute unique articles summary
        unique_articles = self.labels_df['article_id'].nunique()
        total_entries = len(self.labels_df)
        
        print(f"✅ Loaded {total_entries} label entries from {labels_file}")
        print(f"📊 Unique articles: {unique_articles}")
        print(f"📋 Columns: {list(self.labels_df.columns)}")
        
        # Create unique articles dataframe for file mapping
        self._create_unique_articles_summary()
        
        return self.labels_df
    
    def _create_unique_articles_summary(self):
        """Create summary of unique articles with their associated labels."""
        if self.labels_df is None:
            return
            
        # Group by article_id and aggregate label information
        article_groups = []
        
        for article_id, group in self.labels_df.groupby('article_id'):
            # Get label types for this article
            label_types = group['type'].unique().tolist()
            label_count = len(group)
            
            # Get dataset_ids (excluding 'Missing')
            dataset_ids = group[group['dataset_id'] != 'Missing']['dataset_id'].unique().tolist()
            
            article_groups.append({
                'article_id': article_id,
                'label_count': label_count,
                'label_types': label_types,
                'has_primary': 'Primary' in label_types,
                'has_secondary': 'Secondary' in label_types,
                'has_missing': 'Missing' in label_types,
                'dataset_count': len(dataset_ids),
                'dataset_ids': dataset_ids
            })
        
        self.unique_articles_df = pd.DataFrame(article_groups)
        print(f"📈 Created unique articles summary: {len(self.unique_articles_df)} unique articles")

    def inventory_document_paths(self, 
                               pdf_dir: str = "train/PDF", 
                               xml_dir: str = "train/XML") -> pd.DataFrame:
        """
        Create inventory of available PDF and XML files mapped to unique articles.
        
        Args:
            pdf_dir (str): Directory name containing PDF files
            xml_dir (str): Directory name containing XML files  
            
        Returns:
            pd.DataFrame: Document inventory with paths and availability flags
        """
        if self.labels_df is None or self.unique_articles_df is None:
            raise ValueError("Labels must be loaded first. Call load_labels() method.")
        
        # Set up directory paths
        pdf_path = self.data_dir / pdf_dir
        xml_path = self.data_dir / xml_dir
        
        print(f"🔍 Scanning directories:")
        print(f"  📁 PDF: {pdf_path}")
        print(f"  📁 XML: {xml_path}")
        
        # Get available files
        available_pdfs = set()
        available_xmls = set()
        
        if pdf_path.exists():
            for pdf_file in pdf_path.glob("*.pdf"):
                available_pdfs.add(pdf_file.stem)
        
        if xml_path.exists():
            for xml_file in xml_path.glob("*.xml"):
                available_xmls.add(xml_file.stem)
        
        print(f"📊 Found {len(available_pdfs)} PDF files and {len(available_xmls)} XML files")
        
        # Map files to unique articles
        inventory_data = []
        
        for _, row in self.unique_articles_df.iterrows():
            article_id = str(row['article_id'])
            
            # Check file availability
            pdf_available = article_id in available_pdfs
            xml_available = article_id in available_xmls
            has_fulltext = pdf_available or xml_available
            
            # Categorize for conversion workflow
            if pdf_available and xml_available:
                conversion_status = 'both_available'
                conversion_priority = 'none'
            elif pdf_available and not xml_available:
                conversion_status = 'pdf_only'
                conversion_priority = 'high'  # Needs PDF->XML conversion
            elif not pdf_available and xml_available:
                conversion_status = 'xml_only'
                conversion_priority = 'none'  # Unusual but no conversion needed
            else:
                conversion_status = 'missing_both'
                conversion_priority = 'critical'  # Problem case
            
            # Build file paths
            pdf_path_str = str(pdf_path / f"{article_id}.pdf") if pdf_available else None
            xml_path_str = str(xml_path / f"{article_id}.xml") if xml_available else None
            
            inventory_data.append({
                'article_id': article_id,
                'label_count': row['label_count'],
                'label_types': row['label_types'],
                'has_primary': row['has_primary'],
                'has_secondary': row['has_secondary'],
                'has_missing': row['has_missing'],
                'dataset_count': row['dataset_count'],
                'pdf_available': pdf_available,
                'pdf_path': pdf_path_str,
                'xml_available': xml_available,
                'xml_path': xml_path_str,
                'has_fulltext': has_fulltext,
                'conversion_status': conversion_status,
                'conversion_priority': conversion_priority
            })
        
        self.document_inventory = pd.DataFrame(inventory_data)
        
        # Generate file availability summary
        self._generate_file_availability_summary()
        
        return self.document_inventory
    
    def _generate_file_availability_summary(self):
        """Generate comprehensive file availability summary."""
        if self.document_inventory is None:
            return
        
        total_articles = len(self.document_inventory)
        
        # File availability counts
        pdf_available = self.document_inventory['pdf_available'].sum()
        xml_available = self.document_inventory['xml_available'].sum()
        has_fulltext = self.document_inventory['has_fulltext'].sum()
        
        # Conversion workflow categorization
        both_available = len(self.document_inventory[self.document_inventory['conversion_status'] == 'both_available'])
        pdf_only = len(self.document_inventory[self.document_inventory['conversion_status'] == 'pdf_only'])
        xml_only = len(self.document_inventory[self.document_inventory['conversion_status'] == 'xml_only'])
        missing_both = len(self.document_inventory[self.document_inventory['conversion_status'] == 'missing_both'])
        
        self.file_availability_summary = {
            'total_unique_articles': total_articles,
            'pdf_available': pdf_available,
            'xml_available': xml_available,
            'has_fulltext': has_fulltext,
            'both_available': both_available,
            'pdf_only': pdf_only,
            'xml_only': xml_only,
            'missing_both': missing_both,
            'conversion_candidates': pdf_only,  # Articles needing PDF->XML conversion
            'problematic_articles': missing_both
        }
        
        # Print summary
        print(f"\n📋 Document Availability Summary (Unique Articles):")
        print(f"  Total unique articles: {total_articles}")
        print(f"  📄 PDFs available: {pdf_available} ({pdf_available/total_articles*100:.1f}%)")
        print(f"  🔖 XMLs available: {xml_available} ({xml_available/total_articles*100:.1f}%)")
        print(f"  📚 Has full-text: {has_fulltext} ({has_fulltext/total_articles*100:.1f}%)")
        
        print(f"\n🔄 PDF→XML Conversion Analysis:")
        print(f"  ✅ Both PDF & XML: {both_available} ({both_available/total_articles*100:.1f}%)")
        print(f"  🔄 PDF only (needs conversion): {pdf_only} ({pdf_only/total_articles*100:.1f}%)")
        print(f"  📝 XML only: {xml_only} ({xml_only/total_articles*100:.1f}%)")
        print(f"  ❌ Missing both: {missing_both} ({missing_both/total_articles*100:.1f}%)")
        
        if missing_both > 0:
            missing_articles = self.document_inventory[
                self.document_inventory['conversion_status'] == 'missing_both'
            ]['article_id'].tolist()
            print(f"\n⚠️  Articles missing both PDF & XML: {missing_articles[:5]}{'...' if len(missing_articles) > 5 else ''}")

    def get_articles_needing_conversion(self) -> pd.DataFrame:
        """
        Get articles that have PDF but no XML (candidates for PDF->XML conversion).
        
        Returns:
            pd.DataFrame: Articles needing PDF->XML conversion
        """
        if self.document_inventory is None:
            raise ValueError("Must inventory documents first. Call inventory_document_paths() method.")
        
        conversion_candidates = self.document_inventory[
            self.document_inventory['conversion_status'] == 'pdf_only'
        ].copy()
        
        print(f"🔄 Found {len(conversion_candidates)} articles needing PDF→XML conversion")
        
        return conversion_candidates
    
    def get_conversion_summary(self) -> Dict:
        """Get detailed conversion workflow summary."""
        if self.file_availability_summary is None:
            raise ValueError("Must inventory documents first. Call inventory_document_paths() method.")
        
        return self.file_availability_summary.copy()
    
    def export_conversion_candidates(self, output_file: str = "conversion_candidates.csv") -> None:
        """
        Export list of articles needing PDF->XML conversion.
        
        Args:
            output_file (str): Output filename for conversion candidates
        """
        conversion_candidates = self.get_articles_needing_conversion()
        
        if len(conversion_candidates) == 0:
            print("✅ No articles need PDF→XML conversion")
            return
        
        # Select relevant columns for conversion workflow
        # export_df = conversion_candidates[[
            # 'article_id', 'pdf_path', 'xml_path', 'label_count', 'has_primary', 'has_secondary'
        # ]].copy()

        # add column for conversion status
        export_df = self.document_inventory[['article_id', 'pdf_path', 'xml_path', 'label_count', 'has_primary', 'has_secondary','has_missing' ,'conversion_status']].copy()
        export_df["convert_to_xml"] = np.where(export_df["conversion_status"] == "pdf_only", True, False)
        output_path = self.data_dir / output_file
        output_path = Path(str(output_path) + self.TEMP_SUFFIX) # add suffix to output file name
        export_df.to_csv(output_path, index=False)
        
        print(f"📝 Exported {len(conversion_candidates)} conversion candidates to {output_path}")

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
        
        # 1. Check for duplicates (label entries level)
        duplicate_entries = self.labels_df.duplicated().sum()
        duplicate_article_entries = self.labels_df['article_id'].duplicated().sum()
        results['duplicate_entries'] = duplicate_entries
        results['duplicate_article_entries'] = duplicate_article_entries
        
        print(f"🔍 Duplicate Analysis:")
        if duplicate_entries > 0:
            print(f"   ⚠️  Found {duplicate_entries} completely duplicate label entries")
        else:
            print(f"   ✅ No completely duplicate label entries")
            
        print(f"   📊 Articles with multiple labels: {duplicate_article_entries} entries")
        print(f"   📊 Unique articles: {self.labels_df['article_id'].nunique()}")
        
        # 2. Check for null values
        null_counts = self.labels_df.isnull().sum()
        results['null_counts'] = null_counts.to_dict()
        
        print(f"\n🔍 Null value check:")
        for col, null_count in null_counts.items():
            if null_count > 0:
                print(f"   ⚠️  {col}: {null_count} nulls ({null_count/len(self.labels_df)*100:.1f}%)")
            else:
                print(f"   ✅ {col}: No nulls")
        
        # 3. Class balance analysis (label entries level)
        class_counts = self.labels_df['type'].value_counts()
        class_percentages = self.labels_df['type'].value_counts(normalize=True) * 100
        results['class_distribution'] = class_counts.to_dict()
        results['class_percentages'] = class_percentages.to_dict()
        
        print(f"\n📊 Class Distribution (Label Entries):")
        for class_type, count in class_counts.items():
            percentage = class_percentages[class_type]
            print(f"   {class_type}: {count} ({percentage:.1f}%)")
        
        # 4. Unique article analysis
        if self.unique_articles_df is not None:
            print(f"\n📊 Unique Article Analysis:")
            print(f"   Total unique articles: {len(self.unique_articles_df)}")
            print(f"   Articles with Primary labels: {self.unique_articles_df['has_primary'].sum()}")
            print(f"   Articles with Secondary labels: {self.unique_articles_df['has_secondary'].sum()}")
            print(f"   Articles with Missing labels: {self.unique_articles_df['has_missing'].sum()}")
            
            # Multi-label analysis
            multi_label_articles = self.unique_articles_df[self.unique_articles_df['label_count'] > 1]
            print(f"   Articles with multiple labels: {len(multi_label_articles)}")
        
        # 5. Expected distribution check
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
        
        # 6. File availability check (if inventory exists)
        if self.file_availability_summary is not None:
            results['file_availability'] = self.file_availability_summary.copy()
            
            total_articles = self.file_availability_summary['total_unique_articles']
            missing_both = self.file_availability_summary['missing_both']
            
            if missing_both > 0:
                print(f"\n⚠️  Document Availability Issues:")
                print(f"   {missing_both} articles missing both PDF & XML files ({missing_both/total_articles*100:.1f}%)")
            else:
                print(f"\n✅ All articles have at least one file format available")
        
        # 7. Visualization
        if show_plots:
            self._create_class_balance_chart(class_counts)
        
        return results
    
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
        
        plt.title('Label Type Distribution (Label Entries)', fontsize=16, fontweight='bold', pad=20)
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
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary statistics."""
        if self.labels_df is None:
            raise ValueError("Must load labels first.")
        
        stats = {
            # Label entries level
            'total_label_entries': len(self.labels_df),
            'unique_articles': self.labels_df['article_id'].nunique(),
            'unique_datasets': self.labels_df['dataset_id'].nunique(),
            'class_distribution': self.labels_df['type'].value_counts().to_dict(),
        }
        
        # Add unique article stats if available
        if self.unique_articles_df is not None:
            stats.update({
                'articles_with_primary': self.unique_articles_df['has_primary'].sum(),
                'articles_with_secondary': self.unique_articles_df['has_secondary'].sum(), 
                'articles_with_missing': self.unique_articles_df['has_missing'].sum(),
                'multi_label_articles': len(self.unique_articles_df[self.unique_articles_df['label_count'] > 1])
            })
        
        # Add file availability stats if available
        if self.file_availability_summary is not None:
            stats.update({
                'documents_with_fulltext': self.file_availability_summary['has_fulltext'],
                'pdf_availability': self.file_availability_summary['pdf_available'],
                'xml_availability': self.file_availability_summary['xml_available'],
                'conversion_candidates': self.file_availability_summary['conversion_candidates'],
                'problematic_articles': self.file_availability_summary['problematic_articles']
            })
        
        return stats

    def export_missing_files(self, output_file: str = "problematic_articles.txt"):
        """Export list of articles missing both PDF and XML files."""
        if self.document_inventory is None:
            raise ValueError("Must inventory documents first. Call inventory_document_paths() method.")
        
        problematic_articles = self.document_inventory[
            self.document_inventory['conversion_status'] == 'missing_both'
        ]['article_id'].tolist()
        
        if not problematic_articles:
            print("✅ No articles missing both PDF and XML files")
            return
        
        output_path = self.data_dir / output_file
        output_path = Path(str(output_path) + self.TEMP_SUFFIX) # add suffix to output file name
        with open(output_path, 'w') as f:
            f.write("# Articles Missing Both PDF and XML Files\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            f.write(f"# Total problematic articles: {len(problematic_articles)}\n\n")
            for article_id in problematic_articles:
                f.write(f"{article_id}\n")
        
        print(f"📝 Exported {len(problematic_articles)} problematic article IDs to {output_path}")

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

    def get_file_statistics(self) -> Dict:
        """Get detailed file availability statistics for reporting."""
        if self.file_availability_summary is None:
            raise ValueError("Must inventory documents first. Call inventory_document_paths() method.")
        
        return {
            'summary': self.file_availability_summary.copy(),
            'conversion_workflow': {
                'ready_for_processing': self.file_availability_summary['both_available'] + self.file_availability_summary['xml_only'],
                'needs_conversion': self.file_availability_summary['pdf_only'],
                'cannot_process': self.file_availability_summary['missing_both']
            }
        } 