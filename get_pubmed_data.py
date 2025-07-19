from Bio import Entrez, Medline

# putting the email for NCBI for getting pubmed data

Entrez.email = "ajaypunia1503@gmail.com"
Entrez.tool = "PubMedMedicineChatBot"

# research pubmed for medicine related articles

search_term = "Metformin AND dosage AND side effects"
handle = Entrez.esearch(db = "pubmed", term = search_term, retmax = 50)
record = Entrez.read(handle)



# check if any result found
if not record["IdList"]:
    print("No articles found")
    exit()
ids = record["IdList"]
print(f"Found {len(ids)} articles.")


# fetching abstract using those ID's

fetch_handle = Entrez.efetch(db = "pubmed", id = ids , rettype = "medline", retmode = "text")
records = list(Medline.parse(fetch_handle))


# save abstract to the file


with open("medicine_abstracts.txt", "w", encoding = "utf-8") as f:
    for rec in records:
        title = rec.get("TI", "No Title")
        abstract = rec.get("AB","")
        if abstract:
            f.write(f"{title}\n{abstract}\n\n")
print("Abstracts saved to medicine_abstracts.txt")
           