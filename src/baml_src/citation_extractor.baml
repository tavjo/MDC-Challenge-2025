

// Create a function to extract the resume from a string.
function ExtractCitation(document: Document) -> CitationExtractor {
  // Specify a client as provider/model-name
  // you can use custom LLM params with a custom client name from clients.baml like "client CustomHaiku"
  client "CustomReasoner" // Set OPENAI_API_KEY to use this client.
  prompt #"
    Extract dataset citations ONLY from this content. Do not include any other types of citations. Any citation included must specifically mention a dataset.
    ONLY provide the citation, do noy include any other texts. 
    Example:
        {
        "data_citation": "E-GEOD-48278",
        "doc_id": "10.1371_journal.pone.0212669",
        "pages": [
            1,
            3
        ]
    }
    If a citation is found on multiple pages, provide all the pages where it is mentioned as an array. The page number associated with each page of the document is provided below:

    {% for page in document.full_text %}
     --- Page Number {{ loop.index }}:
     {{ page }}
     ---
     {% endfor %}

    {{ ctx.output_format }}
  "#
}



// Test the function with a sample resume. Open the VSCode playground to run this.
test get_citation_entities {
  functions [ExtractCitation]
  args {
    document {
      doi "10.1101_2022.07.21.501061"
      full_text [
                    #"bioRxiv preprint\n\ndoi:\n\nhttps://doi.org/10.1101/2022.07.21.501061\n\n;\n\nthis version posted July 22, 2022.\n\nThe copyright holder for this preprint (which\n\nwas not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made\n\navailable under a\n\nCC-BY-NC 4.0 International license .\n\n1\n\nClimate Ecology as a Driver of Global Breeding Periods in Anurans\n\n2\n\n3\n\nBryan H. Juarez and Lauren A."#,
                    #"
                      .\n\nData availability Repository details and associated metadata for curated samples can be found in Supplementary file 1. MTIII and other element measurement data are in Figure 2\u2014source data 1, and the Rscript used for morphometric analysis is in the DRYAD database (Heintzman et al., 2017). MtDNA genome sequences have been deposited in Genbank under accessions KT168317-KT168336, MF134655- MF134663, and an updated version of JX312727. All mtDNA genome alignments (in NEXUS format) and associated XML and TREE files are in the DRYAD database (Heintzman et al., 2017).
                    "#,
                    #"
                      Raw shot- gun sequence data used for the nuclear genomic analyses and raw shotgun and target enrichment sequence data for TMM 34\u20132518 (francisci holotype) have been deposited in the Short Read Archive (BioProject: PRJNA384940).\n\nNomenclatural act The electronic edition of this article conforms to the requirements of the amended International Code of Zoological Nomenclature, and hence the new name contained herein is available under that Code from the electronic edition of this article. This published work and the nomenclatural act it contains have been registered in ZooBank, the online registration system for the ICZN. The ZooBank LSIDs (Life Science Identifiers) can be resolved and the associated information viewed through any standard web browser by appending the LSID to the prefix \u2018http://zoobank.org/\u2019. The LSID for this publication is: urn:lsid:zoobank.org:pub:8D270E0A-9148-4089-920C-724F07D8DC0B. The electronic edition of this work was published in a journal with an ISSN, and has been archived and is available from the following digital repositories: PubMed Central and LOCKSS.\n\nAcknowledgements We thank the Klondike placer gold mining community of Yukon for their support and providing access to their mines from which many of our Haringtonhippus fossils were recovered.
                    "#
      ]
      n_pages 3
      total_char_length 10000
      total_tokens 1000
      file_hash "1234567890"
      file_path "path/to/file.pdf"
      parsed_timestamp "2025-07-13"
    }
  }
}
