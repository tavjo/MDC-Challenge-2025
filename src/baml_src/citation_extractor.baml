
enum DatasetType {
  PRIMARY @description("Raw or processed data generated as part of the paper, specifically for the study")
  SECONDARY @description("Raw or processed data derived or reused from existing records or published data")
}
class CitationEntity {
    evidence string[] @description("List of evidence from document for the data citation")
    data_citation string @description("Data citation from text (Data repository DOI or accession number. Examples data DOI: https://doi.org/10.5061/dryad.4dj6042. Example accession number: JX123456)")
    dataset_type DatasetType @description("Type of dataset citation")
    @@assert(citation_in_evidence, {{ this.data_citation in (this.evidence | join(' ')) }})
    @@assert(non_empty, {{ this.data_citation|length >= 6 }})
    @@assert(no_spaces_inside, {{ ' ' not in this.data_citation or this.data_citation.startswith('PDB ') }})
    @@assert(no_trailing_punct, {{ this.data_citation[-1] not in ['.', ',', ';', ':', ')'] }})
    @@assert(single_identifier, {{ ',' not in this.data_citation and ' and ' not in this.data_citation and ' to ' not in this.data_citation }})

    // Accept http URLs only if they are DOI URLs OR evidence states a data-availability context
    @@assert(http_requires_dataset_context, {{
      not this.data_citation.startswith('http') or
      'doi.org/10.' in this.data_citation or 'dx.doi.org/10.' in this.data_citation or
      (
        'data availability' in (this.evidence | join(' ') | lower) or
        'data are available' in (this.evidence | join(' ') | lower) or
        'data is available' in (this.evidence | join(' ') | lower) or
        'deposited' in (this.evidence | join(' ') | lower) or
        'repository' in (this.evidence | join(' ') | lower) or
        'dataset' in (this.evidence | join(' ') | lower) or
        'accession' in (this.evidence | join(' ') | lower) or
        'available from' in (this.evidence | join(' ') | lower) or
        'available at' in (this.evidence | join(' ') | lower) or
        'accession number' in (this.evidence | join(' ') | lower)
      )
    }})

    // Unknown 10.x DOIs must either be from dataset registrants or have dataset cues in evidence
    @@assert(unknown_doi_requires_dataset_context, {{
      not this.data_citation.startswith('10.') or
      this.data_citation.startswith('10.5061') or this.data_citation.startswith('10.5281') or
      this.data_citation.startswith('10.1594') or this.data_citation.startswith('10.6084') or
      this.data_citation.startswith('10.25386') or this.data_citation.startswith('10.7910') or
      this.data_citation.startswith('10.15468') or this.data_citation.startswith('10.18112') or
      this.data_citation.startswith('10.5067') or this.data_citation.startswith('10.5063') or
      (
        'repository' in (this.evidence | join(' ') | lower) or
        'dataset' in (this.evidence | join(' ') | lower) or
        'accession' in (this.evidence | join(' ') | lower) or
        'available at' in (this.evidence | join(' ') | lower) or
        'available from' in (this.evidence | join(' ') | lower) or
        'deposited' in (this.evidence | join(' ') | lower) or
        'accession number' in (this.evidence | join(' ') | lower)
      )
    }})

    // Simple reference-like guard (permissive)
    @@assert(not_reference_like_evidence, {{ not (
      'et al.' in (this.evidence | join(' ') | lower) or
      ' vol.' in (this.evidence | join(' ') | lower) or
      ' volume ' in (this.evidence | join(' ') | lower) or
      ' pp.' in (this.evidence | join(' ') | lower)
    )}})

    // Blacklist common article DOI registrants
    @@assert(not_article_doi, {{ not (
      this.data_citation.startswith('10.1016') or
      this.data_citation.startswith('10.1038') or
      this.data_citation.startswith('10.1126') or
      this.data_citation.startswith('10.1002') or
      this.data_citation.startswith('10.1093') or
      this.data_citation.startswith('10.1039') or
      this.data_citation.startswith('10.1364') or
      this.data_citation.startswith('10.1109') or
      this.data_citation.startswith('10.1111') or
      this.data_citation.startswith('10.1145') or
      this.data_citation.startswith('10.1007') or
      this.data_citation.startswith('10.1371') or
      this.data_citation.startswith('10.1098') or
      this.data_citation.startswith('10.1534') or
      this.data_citation.startswith('10.3389') or
      this.data_citation.startswith('10.12688')
    )}})
}

// Create a function to extract the data citations from a list of strings.
function ExtractCitation(document: string[]) -> CitationEntity[] {
  client QwenLocalRR
  prompt #"
    /no_think
    Extract dataset citations only.

    Valid outputs:
    - Dataset DOIs (bare 10.x/... or DOI URLs containing doi.org/10. or dx.doi.org/10.).
    - Accessions: GSE/GSM, SRR/ERR/DRR, PRJNA/PRJEB/PRJDB, E-MTAB-/E-MEXP-, PXD, MSV, EGAD/EGAS, or PDB 1Y2T-style.

    Hard rules:
    - data_citation is the bare identifier only (no labels like "doi:" or "GEO:").
    - One identifier per item (split lists/ranges into separate items).
    - Homepage-only URLs are invalid; dataset webpages (including non‑DOI hosts) are allowed only when the evidence states availability/deposit/accession.
    - Never output the article DOI or any article DOI (journal DOIs such as 10.1016, 10.1038, 10.1126, etc.).
    - Do not return generic database names without IDs.

    Evidence:
    - evidence must contain the identifier verbatim, and data_citation must be an exact substring.

    If no valid dataset DOI or accession is present, return an empty list. Do not guess.

    {{ _.role("user") }}
    {% for text in document %}
    {{ text }}
    {% endfor %}

    {{ ctx.output_format }}

  "#
}



// Test the function with a sample resume. Open the VSCode playground to run this.
test get_citation_entities {
  functions [ExtractCitation]
  args {
      document [
                    #"bioRxiv preprint\n\ndoi:\n\nhttps://doi.org/10.1101/2022.07.21.501061\n\n;\n\nthis version posted July 22, 2022.\n\nThe copyright holder for this preprint (which\n\nwas not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made\n\navailable under a\n\nCC-BY-NC 4.0 International license .\n\n1\n\nClimate Ecology as a Driver of Global Breeding Periods in Anurans\n\n2\n\n3\n\nBryan H. Juarez and Lauren A."#,
                    #".\n\nData availability Repository details and associated metadata for curated samples can be found in Supplementary file 1. MTIII and other element measurement data are in Figure 2\u2014source data 1, and the Rscript used for morphometric analysis is in the DRYAD database (Heintzman et al., 2017). MtDNA genome sequences have been deposited in Genbank under accessions KT168317-KT168336, MF134655- MF134663, and an updated version of JX312727. All mtDNA genome alignments (in NEXUS format) and associated XML and TREE files are in the DRYAD database (Heintzman et al., 2017).
                    "#,
                    #"Raw shot- gun sequence data used for the nuclear genomic analyses and raw shotgun and target enrichment  sequencedata for TMM 34\u20132518 (francisci holotype) have been deposited in the Short Read Archive (BioProject: PRJNA384940).\n\nNomenclatural act The electronic edition of this article conforms to the requirements of the amended International Code of Zoological Nomenclature, and hence the new name contained herein is available under that Code from the electronic edition of this article. This published work and the nomenclatural act it contains have been registered in ZooBank, the online registration system for the ICZN. The ZooBank LSIDs (Life Science Identifiers) can be resolved and the associated information viewed through any standard web browser by appending the LSID to the prefix \u2018http://zoobank.org/\u2019. The LSID for this publication is: urn:lsid:zoobank.org:pub:8D270E0A-9148-4089-920C-724F07D8DC0B. The electronic edition of this work was published in a journal with an ISSN, and has been archived and is available from the following digital repositories: PubMed Central and LOCKSS.\n\nAcknowledgements We thank the Klondike placer gold mining community of Yukon for their support and providing access to their mines from which many of our Haringtonhippus fossils were recovered."#,
                    #"The lineage-specific\n\n534\n\nevolution of aquaporin gene clusters facilitated tetrapod terrestrial adaptation. PLoS One,\n\n535\n\n9, e113686.\n\n536\n\nFisher, M.C. & Garner, T.W.J. (2020). Chytrid fungi and global amphibian declines. Nat. Rev.\n\n537\n\nMicrobiol., 18, 332\u2013343.\n\n538\n\nFreidenburg, L.K. & Skelly, D.K. (2004). Microgeographical variation in thermal preference by an\n\n539\n\namphibian. Ecol. Lett., 7, 369\u2013373.\n\n540\n\nGBIF.org. (2021a). Data from: GBIF Occurrence Download. Available at:\n\n541\n\nhttps://doi.org/10.15468/dl.354f8k. Accessed: 03 October 2021.\n\n542\n\nGBIF.org. (2021b). Data from: GBIF Occurrence Download. Available at:\n\n543\n\nhttps://doi.org/10.15468/dl.pdjqte. Accessed: 03 October 2021.\n\n544\n\nGBIF.org. (2021c). Data from: GBIF Occurrence Download. Available at:\n\n545\n\nhttps://doi.org/10.15468/dl.nbku3v. Accessed: 03 October 2021.\n\n546\n\nGBIF.org. (2021d). Data from: GBIF Occurrence Download. Available at:\n\n547\n\nhttps://doi.org/10.15468/dl.uejpg6. Accessed: 03 October 2021.\n\n548\n\nGiaretta, A.A., Facure, K.G., Sawaya, R.J., Meyer, J.H.D.M. & Chemin, N. (1999)."#
      ]
  }
}
