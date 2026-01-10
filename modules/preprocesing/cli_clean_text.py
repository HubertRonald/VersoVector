from modules.preprocesing.clean_text import preprocess

if __name__ == "__main__":
    sample_text = "This is a sample text! Visit http://example.com for more info."
    processed_text = preprocess(sample_text)
    print("Original Text:", sample_text)
    print("Processed Text:", processed_text)
    