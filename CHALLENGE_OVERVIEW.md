# Make Data Count - Finding Data References Challenge

## Overview

Your efforts can help Make Data Count (MDC). Scientific data are critically undervalued even though they provide the basis for discoveries and innovations. We aim to improve our understanding of the links between scientific papers and the data used in those studies—what, when, and how they are mentioned. This will help to establish the value and impact of open scientific data for reuse. The current version of the MDC Data Citation Corpus is an aggregation of links between data and papers, but these links are incomplete: they only cover a fraction of the literature and do not provide context on how the data were used. The outcome of this competition is a highly performant model that will continuously run on scientific literature to automate the addition of high quality and contextualized data-to-paper connections in to the MDC Data Citation Corpus.

## Goal of the Competition

You will identify all the data citations (references to research data) from the full text of scientific literature and tag the type of citation (primary or secondary):

- **Primary** - raw or processed data generated as part of the paper, specifically for the study
- **Secondary** - raw or processed data derived or reused from existing records or published data

## Dataset Structure

### Data Overview
The dataset contains scientific papers from the Europe PMC open access subset, available in both PDF and XML formats. Participants will extract research data references from these papers and classify them by citation type.

### Data Identifiers
- **DOIs**: Used for all papers and some datasets (format: `https://doi.org/[prefix]/[suffix]`)
  - Example: `https://doi.org/10.1371/journal.pone.0303785`
- **Accession IDs**: Used for datasets from specific repositories
  - Examples: `GSE12345` (GEO), `PDB 1Y2T` (Protein Data Bank), `E-MEXP-568` (ArrayExpress)

### Directory Structure
```
MDC-Challenge-2025/Data/
├── train/
│   ├── PDF/ (524 files)
│   └── XML/ (400 files - ~75% coverage)
├── test/
│   ├── PDF/ (30 files in sample, ~2,600 in full test)
│   └── XML/ (25 files in sample)
├── train_labels.csv
└── sample_submission.csv
```

## Context

Make Data Count (MDC) is a global, community-driven initiative focused on establishing open standardized metrics for the evaluation and reward of research data reuse and impact. Through both advocacy and infrastructure projects, Make Data Count facilitates the recognition of data as a primary research output, promoting data sharing and reuse across data communities.

Highlighting and valuing data contributions will lead to a more collaborative, transparent, and efficient science, ultimately driving innovation and progress. To assure that this can happen, we need to connect and contextualize data, their relationship to papers, and their reuse.

## Why This is Not YET a Solved Problem

Studies have shown that most research data remain "uncited" (~86%) (Peters, I., Kraker, P., Lex, E. et al., 2016, https://doi.org/10.1007/s11192-016-1887-4) in the current data citation system which makes it very hard to identify and record them. In addition, references to data are harder to programmatically identify because of the many ways they are mentioned. For example, authors may:

- Provide a full description of the data in the methods section
- Indirectly mention it elsewhere in the text
- Provide a formal citation in the reference list
- Use variable language when describing data relationships ("publicly available", "obtained from", etc.)

## Evaluation

The competition uses **F1-Score** as the primary metric. The F1-score measures accuracy using precision (p) and recall (r):

- **Precision**: ratio of true positives (tp) to all predicted positives (tp + fp)
- **Recall**: ratio of true positives to all actual positives (tp + fn)

**F1 Score Formula:**
```
F1 = 2 * (precision * recall) / (precision + recall)
```

The F1 metric weights recall and precision equally, favoring algorithms that maximize both simultaneously rather than excelling at one while performing poorly at the other.

## Submission Requirements

### Format
Predictions must form unique tuples of `(article_id, dataset_id, type)`:
- If an article contains multiple references of the same `dataset_id` and `type`, predict only once
- Include only articles containing data references
- Articles with no data references should NOT be included (will be penalized as false positives)
- Convert all DOIs to full format: `https://doi.org/[prefix]/[suffix]`

### File Structure
```csv
row_id,article_id,dataset_id,type
0,10.1002_cssc.202201821,https://doi.org/10.5281/zenodo.7074790,Primary
1,10.1002_esp.5090,CHEMBL1097,Secondary
```

## Timeline

- **June 11, 2025** - Start Date
- **September 2, 2025** - Entry Deadline & Team Merger Deadline
- **September 9, 2025** - Final Submission Deadline

*All deadlines are at 11:59 PM UTC*

## Prizes

- **1st Place**: $40,000
- **2nd Place**: $20,000
- **3rd Place**: $17,000
- **4th Place**: $13,000
- **5th Place**: $10,000

*Competition prizes are kindly sponsored by The Navigation Fund and Chan Zuckerberg Initiative.*

## Technical Requirements

### Code Submission
- CPU/GPU Notebook ≤ 9 hours run-time
- Internet access disabled
- Freely & publicly available external data allowed (including pre-trained models)
- Submission file must be named `submission.csv`

## Potential Impact

The winning Kaggle model will enable MDC to update, release, and maintain an open, comprehensive and high quality set of references to data in scientific papers. The subsequent corpus will be made freely available for use by research communities, allowing for:

- Development of better tools
- Better understanding of how data are reused
- Improved ways of capturing broader researcher outputs
- A shift towards valuing data

## About Make Data Count

Make Data Count is a global, community-led initiative focused on the development of open data assessment metrics. Driven by partners at academic institutions and non-profit research infrastructure providers, MDC has worked for over a decade to develop technical infrastructure and standardized approaches for assessing data usage, produce evidence-based studies on researcher behavior in citing and using data, and to drive a community of practice around responsible and meaningful evaluation of data reuse.

The competition is sponsored by MDC's fiscal home, DataCite International Data Citation Initiative e.V, with prize funds from The Navigation Fund and Chan Zuckerberg Initiative.

## Getting Started

1. **Explore Training Data**: Review files in `Data/train/` directory
2. **Understand Labels**: Study `train_labels.csv` for citation type examples
3. **Run EDA**: Use notebooks in `notebooks/` directory for exploratory analysis
4. **Develop Solution**: Build your models in the `src/` directory
5. **Test & Store**: Validate models and save in `models/` directory

## Citation

Make Data Count, Maggie Demkin, and Walter Reade. Make Data Count - Finding Data References. https://kaggle.com/competitions/make-data-count-finding-data-references, 2025. Kaggle. 