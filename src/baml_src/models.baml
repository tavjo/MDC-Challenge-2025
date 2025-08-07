class CitationEntity {
    evidence string[] @description("List of evidence from the text for the data citation")
    data_citation string @description("Data citation from text")
    // doc_id string? @description("DOI of the document where the data citation is found")
    // pages int[]? @description("List of page numbers where the data citation is mentioned.")
}

// class CitationExtractor {
//     citation_entities CitationEntity[] @description("List of dataset citation entities found in the document")
//     evidence string @description("Evidence in the text for the choosing the extracted dataset citation entities")
// }

class Document {
    doi string @description("DOI of the document")
    has_dataset_citation bool? @description("Whether the document has 1 or more dataset citation")
    full_text string[] @description("Full text of the document")
    total_char_length int @description("Total number of characters in the document")
    parsed_timestamp string @description("Timestamp of when the document was parsed")
    total_tokens int @description("Total number of tokens in the document")
    avg_tokens_per_chunk float? @description("Average number of tokens per chunk in the document")
    file_hash string @description("Hash of the document file")
    file_path string @description("Path to the document file")
    n_pages int @description("Number of pages in the document")
    citation_entities CitationEntity[]? @description("List of citation entities found in the document")
}

