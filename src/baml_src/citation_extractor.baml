
enum DatasetType {
  PRIMARY @description("Raw or processed data generated as part of the paper, specifically for the study")
  SECONDARY @description("Raw or processed data derived or reused from existing records or published data")
}
class CitationEntity {
    evidence string[] @description("List of evidence from document for the data citation")
    dataset_type DatasetType @description("Type of dataset citation")
    data_citation string @description("Data citation from text")
}

// Create a function to extract the data citations from a list of strings.
function ExtractCitation(document: string[]) -> CitationEntity[] {
  client QwenLocal
  prompt #"
    /no_think

    You are a research assistant tasked with extracting dataset citations from a text. Extract ALL dataset citations from the document.
    The evidence should be the snippet of text from the document that contains the dataset citation.

    Use context provided in the document to determine if a citation is a data citation such as "Dataset deposited in ...", "Data retrieved from ...", "Data accessed from ...","Data available " etc.
    
    Report dataset citations EXACTLY as they appear in the text but do not include the surrounding text itself. Don't add the reporsitory name unless it already a part of the data citation. 
    
    Return only dataset repository DOIs (Dryad, Zenodo, Figshare, PANGAEA, etc.) and accession IDs (GEO/SRA/ENA/GenBank/PRJNA/PRJEB/PXD/E-* etc.). Do NOT return article/preprint/journal DOIs (e.g., bioRxiv/medRxiv, publisher DOIs). Examine context surrounding DOIs to judge whether they are data citations or not.

    Never make up a data citation. If you cannot find a data citation, return an empty list.

    When accession numbers are grouped together to form a range such as in this example: JX312727-JX312728 (example accession range), you must report each accession number in the range as a separate CitationEntity:

    Example:
    [
    {
      "evidence": ["Data deposited in GenBank under accessions JX312727-JX312728."],
      "dataset_type": "PRIMARY",
      "data_citation": "JX312727",
    },
    {
      "evidence": ["Data deposited in GenBank under accessions JX312727-JX312728."],
      "dataset_type": "PRIMARY",
      "data_citation": "JX312728",
    },
    ]
    IMPORTANT: Return each accession number or dataset repository DOI as separate data citations.

    DO NOT RETURN ANYTHING THAT IS NOT A DATA CITATION (i.e. accession number or repository DOI). 

    {{ ctx.output_format }}

    {{ _.role("user") }}

    {% for text in document %}
    --- {{ text }}
    {% endfor %}
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
