import pandas as pd
import numpy as np
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import logging
from datetime import datetime
import multiprocessing as mp
import torch
import glob
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler(f'deduplication_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
       logging.StreamHandler()
   ]
)
logger = logging.getLogger(__name__)

class MedicalComplianceDeduplicator:
   def __init__(self,
                embedding_model: str = "all-MiniLM-L6-v2",
                semantic_threshold: float = 0.85,
                use_gpu: bool = True,
                n_workers: int = None):

       self.embedding_model_name = embedding_model
       self.semantic_threshold = semantic_threshold
       self.use_gpu = use_gpu and torch.cuda.is_available()
       self.n_workers = n_workers or mp.cpu_count()

       self.device = 'cuda' if self.use_gpu else 'cpu'
       logger.info(f"Loading embedding model: {embedding_model} on {self.device}")
       self.model = SentenceTransformer(embedding_model, device=self.device)

       self.original_data = None
       self.deduplicated_data = None
       self.duplicate_groups = defaultdict(list)
       self.statistics = {}

       logger.info(f"Initialized with {self.n_workers} workers, GPU: {self.use_gpu}")

   def get_all_csv_files(self, data_folder: str) -> List[str]:
       csv_files = glob.glob(f"{data_folder}/*.csv")
       csv_files.sort()
       logger.info(f"Found {len(csv_files)} CSV files in {data_folder}")
       for file in csv_files:
           logger.info(f"  - {Path(file).name}")
       return csv_files

   def load_and_combine_data(self, file_paths: List[str]) -> pd.DataFrame:
       logger.info("Loading and combining data files...")
       combined_data = []

       for file_path in file_paths:
           path = Path(file_path)
           if not path.exists():
               logger.error(f"File not found: {file_path}")
               continue

           try:
               df = pd.read_csv(file_path)
               df['source_file'] = path.name
               combined_data.append(df)
               logger.info(f"Loaded {len(df)} rows from {path.name}")
           except Exception as e:
               logger.error(f"Error loading {file_path}: {e}")

       if not combined_data:
           raise ValueError("No data files could be loaded")

       result = pd.concat(combined_data, ignore_index=True)
       logger.info(f"Combined dataset: {len(result)} total rows")
       return result

   def preprocess_text(self, text: str) -> str:
       if pd.isna(text):
           return ""

       text = str(text).strip()
       text = re.sub(r'\s+', ' ', text)
       text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', ' ', text)
       return text.lower()

   def compute_content_hash(self, row: pd.Series, columns: List[str]) -> str:
       content = ""
       for col in columns:
           if col in row.index:
               content += self.preprocess_text(str(row[col]))
       return hashlib.sha256(content.encode()).hexdigest()

   def find_exact_duplicates(self, df: pd.DataFrame,
                           content_columns: List[str]) -> Dict[str, List[int]]:
       logger.info("Finding exact duplicates...")
       hash_groups = defaultdict(list)

       for idx, row in df.iterrows():
           content_hash = self.compute_content_hash(row, content_columns)
           hash_groups[content_hash].append(idx)

       duplicates = {h: indices for h, indices in hash_groups.items()
                    if len(indices) > 1}

       logger.info(f"Found {len(duplicates)} exact duplicate groups")
       return duplicates

   def compute_embeddings(self, texts: List[str]) -> np.ndarray:
       logger.info(f"Computing embeddings for {len(texts)} texts...")

       if len(texts) > 50000:
           batch_size = 16
       elif len(texts) > 10000:
           batch_size = 32
       else:
           batch_size = 64

       logger.info(f"Using batch size: {batch_size}")
       embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)
       return embeddings

   def find_optimal_threshold(self, df: pd.DataFrame,
                            content_columns: List[str],
                            test_thresholds: List[float] = None,
                            sample_size: int = 10000) -> Tuple[float, Dict]:

       if test_thresholds is None:
           test_thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

       logger.info("Finding optimal semantic similarity threshold...")
       logger.info(f"Testing thresholds: {test_thresholds}")

       if len(df) > sample_size:
           logger.info(f"Large dataset detected ({len(df)} rows). Using sample of {sample_size} rows for threshold optimization.")
           sample_df = df.sample(n=sample_size, random_state=42)
       else:
           sample_df = df

       logger.info(f"Using {len(sample_df)} rows for threshold analysis")

       combined_texts = []
       for _, row in sample_df.iterrows():
           combined_text = ' '.join([
               self.preprocess_text(str(row[col])) for col in content_columns
               if col in row.index
           ])
           combined_texts.append(combined_text)

       embeddings = self.compute_embeddings(combined_texts)
       similarity_matrix = cosine_similarity(embeddings)

       threshold_analysis = {}

       for threshold in test_thresholds:
           logger.info(f"Testing threshold: {threshold}")

           semantic_groups = defaultdict(list)
           processed_indices = set()

           for i in range(len(similarity_matrix)):
               if i in processed_indices:
                   continue

               similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
               similar_indices = [idx for idx in similar_indices if idx not in processed_indices]

               if len(similar_indices) > 1:
                   group_key = f"semantic_group_{len(semantic_groups)}"
                   semantic_groups[group_key] = similar_indices
                   processed_indices.update(similar_indices)

           total_duplicates = sum(len(group) for group in semantic_groups.values())
           unique_duplicate_rows = len(processed_indices)
           data_retention = ((len(sample_df) - unique_duplicate_rows) / len(sample_df)) * 100
           duplicate_percentage = (unique_duplicate_rows / len(sample_df)) * 100

           threshold_analysis[threshold] = {
               'groups': len(semantic_groups),
               'total_duplicates': total_duplicates,
               'unique_duplicate_rows': unique_duplicate_rows,
               'data_retention_pct': data_retention,
               'duplicate_percentage': duplicate_percentage,
               'final_dataset_size': len(sample_df) - unique_duplicate_rows
           }

           logger.info(f"  Threshold {threshold}: {len(semantic_groups)} groups, "
                      f"{unique_duplicate_rows} duplicates ({duplicate_percentage:.1f}%), "
                      f"data retention: {data_retention:.1f}%")

       optimal_threshold = None
       best_score = -1

       for threshold, stats in threshold_analysis.items():
           if 10 <= stats['duplicate_percentage'] <= 80:
               score = stats['data_retention_pct'] * 0.7 + (100 - stats['duplicate_percentage']) * 0.3

               if score > best_score:
                   best_score = score
                   optimal_threshold = threshold

       if optimal_threshold is None:

           optimal_threshold = max(threshold_analysis.keys(),
                                 key=lambda t: threshold_analysis[t]['data_retention_pct'])

       logger.info(f"Optimal threshold selected: {optimal_threshold}")
       logger.info(f"  Data retention: {threshold_analysis[optimal_threshold]['data_retention_pct']:.1f}%")
       logger.info(f"  Duplicates removed: {threshold_analysis[optimal_threshold]['duplicate_percentage']:.1f}%")

       return optimal_threshold, threshold_analysis

   def find_semantic_duplicates(self, df: pd.DataFrame,
                              content_columns: List[str]) -> Dict[str, List[int]]:
       logger.info("Finding semantic duplicates...")

       combined_texts = []
       for _, row in df.iterrows():
           combined_text = ' '.join([
               self.preprocess_text(str(row[col])) for col in content_columns
               if col in row.index
           ])
           combined_texts.append(combined_text)

       embeddings = self.compute_embeddings(combined_texts)

       logger.info("Computing similarity matrix...")
       similarity_matrix = cosine_similarity(embeddings)

       semantic_groups = defaultdict(list)
       processed_indices = set()

       for i in range(len(similarity_matrix)):
           if i in processed_indices:
               continue

           similar_indices = np.where(similarity_matrix[i] >= self.semantic_threshold)[0]
           similar_indices = [idx for idx in similar_indices if idx not in processed_indices]

           if len(similar_indices) > 1:
               group_key = f"semantic_group_{len(semantic_groups)}"
               semantic_groups[group_key] = similar_indices
               processed_indices.update(similar_indices)

       logger.info(f"Found {len(semantic_groups)} semantic duplicate groups")
       return semantic_groups

   def analyze_duplicates(self, df: pd.DataFrame, duplicate_groups: Dict) -> Dict:
       total_duplicates = sum(len(group) for group in duplicate_groups.values())
       duplicate_indices = set()
       for group in duplicate_groups.values():
           duplicate_indices.update(group)

       stats = {
           'total_groups': len(duplicate_groups),
           'total_duplicates': total_duplicates,
           'unique_duplicate_rows': len(duplicate_indices),
           'duplicate_percentage': (len(duplicate_indices) / len(df)) * 100 if len(df) > 0 else 0,
           'avg_group_size': total_duplicates / len(duplicate_groups) if duplicate_groups else 0,
           'max_group_size': max(len(group) for group in duplicate_groups.values()) if duplicate_groups else 0
       }
       return stats

   def remove_duplicates(self, df: pd.DataFrame,
                        duplicate_groups: Dict,
                        strategy: str = 'keep_first') -> pd.DataFrame:
       indices_to_remove = set()

       for group_id, indices in duplicate_groups.items():
           if len(indices) <= 1:
               continue

           if strategy == 'keep_first':
               indices_to_remove.update(indices[1:])
           elif strategy == 'keep_best':
               best_idx = indices[0]
               best_completeness = 0

               for idx in indices:
                   completeness = df.loc[idx].count() / len(df.columns)
                   if completeness > best_completeness:
                       best_completeness = completeness
                       best_idx = idx

               indices_to_remove.update([idx for idx in indices if idx != best_idx])

       result_df = df.drop(index=list(indices_to_remove)).reset_index(drop=True)

       logger.info(f"Removed {len(indices_to_remove)} duplicate rows")
       logger.info(f"Resulting dataset: {len(result_df)} rows")

       return result_df

   def create_duplicate_report(self, duplicate_groups: Dict,
                             df: pd.DataFrame,
                             report_type: str) -> str:
       report = [f"\n=== {report_type.upper()} DUPLICATES REPORT ===\n"]

       if not duplicate_groups:
           report.append("No duplicates found.\n")
           return '\n'.join(report)

       stats = self.analyze_duplicates(df, duplicate_groups)

       report.append(f"Total duplicate groups: {stats['total_groups']}")
       report.append(f"Total duplicate rows: {stats['unique_duplicate_rows']}")
       report.append(f"Percentage of dataset: {stats['duplicate_percentage']:.2f}%")
       report.append(f"Average group size: {stats['avg_group_size']:.2f}")
       report.append(f"Maximum group size: {stats['max_group_size']}")
       report.append("")

       for i, (group_id, indices) in enumerate(list(duplicate_groups.items())[:3]):
           report.append(f"Group {i+1} ({len(indices)} items):")
           for j, idx in enumerate(indices[:2]):
               prompt = str(df.loc[idx, 'prompt'])[:100] + "..."
               report.append(f"  [{idx}] {prompt}")
           if len(indices) > 2:
               report.append(f"  ... and {len(indices)-2} more items")
           report.append("")

       if len(duplicate_groups) > 3:
           report.append(f"... and {len(duplicate_groups)-3} more groups")

       return '\n'.join(report)

   def create_threshold_analysis_report(self, threshold_analysis: Dict, optimal_threshold: float) -> str:
       report = ["\n=== THRESHOLD ANALYSIS REPORT ===\n"]

       report.append(f"Optimal threshold selected: {optimal_threshold}")
       report.append("")
       report.append("Threshold comparison:")
       report.append("Threshold | Groups | Duplicates | Data Retention | Final Size")
       report.append("-" * 60)

       for threshold in sorted(threshold_analysis.keys()):
           stats = threshold_analysis[threshold]
           report.append(f"   {threshold:<6} | {stats['groups']:<6} | "
                        f"{stats['unique_duplicate_rows']:<10} | "
                        f"{stats['data_retention_pct']:<13.1f}% | "
                        f"{stats['final_dataset_size']:<10}")

       return '\n'.join(report)

   def visualize_duplicates(self, stats: Dict, output_dir: str = "."):
       output_path = Path(output_dir)
       output_path.mkdir(exist_ok=True)

       fig, axes = plt.subplots(1, 2, figsize=(12, 5))
       fig.suptitle('Medical Compliance Data Deduplication Analysis', fontsize=14)

       categories = ['Original', 'After Exact\nDedup', 'After Semantic\nDedup']
       counts = [
           stats['original_count'],
           stats['after_exact_dedup'],
           stats['after_semantic_dedup']
       ]

       axes[0].bar(categories, counts, color=['blue', 'orange', 'green'])
       axes[0].set_title('Dataset Size After Deduplication')
       axes[0].set_ylabel('Number of Rows')

       for i, count in enumerate(counts):
           axes[0].text(i, count + max(counts)*0.01, f'{count:,}',
                       ha='center', va='bottom')

       duplicate_counts = [
           stats['exact_duplicates_count'],
           stats['semantic_duplicates_count']
       ]

       if sum(duplicate_counts) > 0:
           axes[1].pie(duplicate_counts, labels=['Exact', 'Semantic'],
                      autopct='%1.1f%%', startangle=90)
           axes[1].set_title('Distribution of Duplicate Types')
       else:
           axes[1].text(0.5, 0.5, 'No duplicates found',
                       ha='center', va='center', transform=axes[1].transAxes)
           axes[1].set_title('No Duplicates Found')

       plt.tight_layout()
       plt.savefig(output_path / 'deduplication_analysis.png', dpi=300, bbox_inches='tight')
       plt.close()

       logger.info(f"Visualization saved to {output_path / 'deduplication_analysis.png'}")

   def run_complete_deduplication(self,
                                file_paths: List[str],
                                content_columns: List[str] = None,
                                output_file: str = "deduplicated_medical_compliance.csv",
                                create_report: bool = True,
                                optimize_threshold: bool = True) -> pd.DataFrame:

       if content_columns is None:
           content_columns = ['prompt', 'response', 'source_input']

       df = self.load_and_combine_data(file_paths)
       self.original_data = df.copy()

       threshold_analysis = None
       if optimize_threshold:
           logger.info("Step 0/4: Finding optimal threshold...")
           optimal_threshold, threshold_analysis = self.find_optimal_threshold(df, content_columns)
           self.semantic_threshold = optimal_threshold

       self.statistics = {
           'original_count': len(df),
           'content_columns': content_columns,
           'semantic_threshold': self.semantic_threshold
       }

       step_num = "1/4" if optimize_threshold else "1/3"
       logger.info(f"Step {step_num}: Finding exact duplicates...")
       exact_duplicates = self.find_exact_duplicates(df, content_columns)
       df = self.remove_duplicates(df, exact_duplicates, strategy='keep_first')

       exact_stats = self.analyze_duplicates(self.original_data, exact_duplicates)
       self.statistics.update({
           'exact_duplicates_count': exact_stats['unique_duplicate_rows'],
           'after_exact_dedup': len(df)
       })

       if create_report:
           self.duplicate_groups['exact'] = exact_duplicates
           logger.info(self.create_duplicate_report(exact_duplicates, self.original_data, "exact"))

       step_num = "2/4" if optimize_threshold else "2/3"
       logger.info(f"Step {step_num}: Finding semantic duplicates...")
       semantic_duplicates = self.find_semantic_duplicates(df, content_columns)
       df = self.remove_duplicates(df, semantic_duplicates, strategy='keep_best')

       semantic_stats = self.analyze_duplicates(df, semantic_duplicates)
       self.statistics.update({
           'semantic_duplicates_count': semantic_stats['unique_duplicate_rows'],
           'after_semantic_dedup': len(df)
       })

       if create_report:
           self.duplicate_groups['semantic'] = semantic_duplicates
           logger.info(self.create_duplicate_report(semantic_duplicates, self.original_data, "semantic"))

           if threshold_analysis:
               logger.info(self.create_threshold_analysis_report(threshold_analysis, self.semantic_threshold))

       step_num = "3/4" if optimize_threshold else "3/3"
       logger.info(f"Step {step_num}: Saving results...")
       self.deduplicated_data = df

       output_path = Path(output_file)
       df.to_csv(output_path, index=False)
       logger.info(f"Deduplicated data saved to {output_path}")

       if create_report:
           self.create_comprehensive_report(output_path.parent, threshold_analysis)
           self.visualize_duplicates(self.statistics, output_path.parent)

       total_removed = self.statistics['original_count'] - len(df)
       removal_pct = (total_removed / self.statistics['original_count']) * 100

       logger.info(f"\n=== DEDUPLICATION SUMMARY ===")
       logger.info(f"Original dataset: {self.statistics['original_count']:,} rows")
       logger.info(f"Final dataset: {len(df):,} rows")
       logger.info(f"Total removed: {total_removed:,} rows ({removal_pct:.2f}%)")
       logger.info(f"Data reduction: {removal_pct:.1f}%")
       logger.info(f"Optimal threshold used: {self.semantic_threshold}")

       return df

   def create_comprehensive_report(self, output_dir: str = ".", threshold_analysis: Dict = None):
       output_path = Path(output_dir)
       report_file = output_path / f"deduplication_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

       with open(report_file, 'w', encoding='utf-8') as f:
           f.write("MEDICAL COMPLIANCE DATA DEDUPLICATION REPORT\n")
           f.write("=" * 50 + "\n\n")

           f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
           f.write(f"Embedding Model: {self.embedding_model_name}\n")
           f.write(f"Semantic Threshold: {self.semantic_threshold}\n\n")

           if threshold_analysis:
               f.write("THRESHOLD OPTIMIZATION ANALYSIS\n")
               f.write("-" * 35 + "\n")
               f.write(f"Optimal threshold: {self.semantic_threshold}\n")
               f.write("Threshold comparison:\n")
               f.write("Threshold | Groups | Duplicates | Data Retention | Final Size\n")
               f.write("-" * 60 + "\n")

               for threshold in sorted(threshold_analysis.keys()):
                   stats = threshold_analysis[threshold]
                   f.write(f"   {threshold:<6} | {stats['groups']:<6} | "
                          f"{stats['unique_duplicate_rows']:<10} | "
                          f"{stats['data_retention_pct']:<13.1f}% | "
                          f"{stats['final_dataset_size']:<10}\n")
               f.write("\n")

           f.write("OVERALL STATISTICS\n")
           f.write("-" * 20 + "\n")
           f.write(f"Original dataset size: {self.statistics['original_count']:,} rows\n")
           f.write(f"Final dataset size: {self.statistics['after_semantic_dedup']:,} rows\n")
           total_removed = self.statistics['original_count'] - self.statistics['after_semantic_dedup']
           removal_pct = (total_removed / self.statistics['original_count']) * 100
           f.write(f"Total rows removed: {total_removed:,} ({removal_pct:.2f}%)\n\n")

           f.write("DEDUPLICATION BREAKDOWN\n")
           f.write("-" * 25 + "\n")
           f.write(f"After exact deduplication: {self.statistics['after_exact_dedup']:,} rows\n")
           f.write(f"After semantic deduplication: {self.statistics['after_semantic_dedup']:,} rows\n\n")

           for dup_type, groups in self.duplicate_groups.items():
               f.write(self.create_duplicate_report(groups, self.original_data, dup_type))
               f.write("\n" + "="*50 + "\n")

       logger.info(f"Comprehensive report saved to {report_file}")

def main():

   DATA_FOLDER = "data"
   OUTPUT_FILE = "medical_compliance_deduplicated.csv"
   CONTENT_COLUMNS = ['prompt', 'response', 'source_input']

   try:
       deduplicator = MedicalComplianceDeduplicator(
           embedding_model="all-MiniLM-L6-v2",
           semantic_threshold=0.85,
           use_gpu=True,
           n_workers=mp.cpu_count()
       )

       INPUT_FILES = deduplicator.get_all_csv_files(DATA_FOLDER)

       deduplicated_df = deduplicator.run_complete_deduplication(
           file_paths=INPUT_FILES,
           content_columns=CONTENT_COLUMNS,
           output_file=OUTPUT_FILE,
           create_report=True,
           optimize_threshold=True
       )

       print(f"\nDeduplication completed successfully!")
       print(f"Output file: {OUTPUT_FILE}")
       print(f"Total files processed: {len(INPUT_FILES)}")
       print(f"Original: {deduplicator.statistics['original_count']:,} rows")
       print(f"Final: {len(deduplicated_df):,} rows")
       print(f"Reduction: {((deduplicator.statistics['original_count'] - len(deduplicated_df)) / deduplicator.statistics['original_count'] * 100):.1f}%")
       print(f"Optimal threshold used: {deduplicator.semantic_threshold}")

   except Exception as e:
       logger.error(f"Deduplication failed: {e}")
       raise

if __name__ == "__main__":
   main()