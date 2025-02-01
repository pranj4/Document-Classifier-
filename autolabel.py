def get_label_from_foldername(folder_name):
    if "invoice" in folder_name.lower():
        return "invoice"
    elif "budget" in folder_name.lower():
        return "budget"
    elif "email" in folder_name.lower():
        return "email"
    else:
        return "other"

# Example
print(get_label_from_foldername(r"C:\Users\91800\OneDrive\Desktop\NLP\data\docs-sm\invoice\01398469.jpg"))  # Should return 'invoice'
print(get_label_from_foldername(r"C:\Users\91800\OneDrive\Desktop\NLP\data\docs-sm\budget\0000076288.jpg"))  # Should return 'budget'